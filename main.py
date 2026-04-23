import torch.multiprocessing as tmx
from basic.dataset import *
from basic.utils import set_seed, AvgMeter
from attack.dlg_attack import perform_dlg
from argparse import Namespace
from torch.utils.tensorboard import SummaryWriter
import yaml
import time
import os
import pickle as pk
import torch
from fl.client import Client, params_norm, dp_scale_laplace, get_delta_norm_by_eps
from fl.server import Server
from basic.config import device, inversefed_setup
import inversefed
from attack.dlg_utils import calculate_ssim
import traceback

def fed_init(args, tb_writer, ds_name, shuffle):
    """new version based on InvGrad Data Processing"""

    # 分好了训练集测试集。还有服务器和客户端。
    # data_path = '/data/Xu_data/FedEM/fednfl-master/data'    # CIFAR-100
    data_path = '/data2/liym/Projects/FedPAC/data'           # 通用数据集
    # data_path = '/data/Xu_data/FedEM2/fednfl-master/data/tiny-imagenet-200'   # Tiny-imagenet

    # 1.init dataset
    if ds_name == "mnist":
        arch = 'LeNetZhuMNIST'   # TODO 这里是专门为MNIST设计的网络模型，是LeNetZhuMNIST
        trainset, testset = inversefed.build_mnist(data_path, False, True)
    elif ds_name == "fmnist":
        arch = 'LeNetZhuMNIST'
        trainset, testset = inversefed.build_fmnist(data_path, False, True)
    elif ds_name == "emnist":
        arch = 'LeNetZhuMNIST'
        trainset, testset = inversefed.build_emnist(data_path, False, True)
    elif ds_name == "cifar10":
        arch = 'ResNet18'
        trainset, testset = inversefed.build_cifar10(data_path, False, True)
    elif ds_name == "cifar100":
        # arch = 'ConvNet64'
        arch = 'ResNet50'
        trainset, testset = inversefed.build_cifar100(data_path, False, True)
    elif ds_name == "tiny":
        # arch = 'ConvNet64'
        arch = 'ResNet50'
        trainset, testset = inversefed.build_tiny_imagenet(data_path, False, True)
    else:
        raise Exception("Does not support for dataset:{args.dataset }.")
    
    # 2.client data partition
    trainsets, valsets, testsets = easy_data_partition(args.n_clients, trainset, testset, shuffle,
                                                           n_train_data_per_client = args.data_per_client)  # 1000

    # 3. init client & server object
    clients = []
    for i in range(args.n_clients):
        client = Client(i, ds_name, arch, trainsets[i], valsets[i], testsets[i], shuffle=args.shuffle,
                        apply_distortion=args.nfl.apply_distortion,
                        distortion_iter=args.nfl.distortion_iter, local_batch_iter=args.local_batch_iter,
                        model_optim=args.model_optim, zeta=args.nfl.zeta, lr=args.lr,
                        bs=args.batch_size, wd=args.weight_decay, le=args.local_epoch,
                        tb_writer=tb_writer,use_rp=args.use_rp, rp_ratio=args.rp_ratio, rp_eps = args.rp_eps)
        
        clients.append(client)
    server = Server(clients, ds_name, arch, args.checkpoint_dir,use_rp=args.use_rp, rp_ratio=args.rp_ratio, rp_eps = args.rp_eps)  # 这里服务器的模型也要做和客户端相应的处理
    return clients, server


def get_nfl_bounds(args, model):
    if args.nfl.privacy == 'nfl' or args.nfl.privacy == 'barre':
        # 参考FedNFL的公式2，根据eps计算l，及失真程度Δ，然后根据失真程度确定上下界。
        l = get_delta_norm_by_eps(args.nfl.eps, args.nfl.D, args.nfl.ca, args.nfl.c0, args.nfl.dlg_iter)
        # u = 2*clip
        u = 2*l
    else:
        sigma_dp = dp_scale_laplace(args.nfl.eps, args.nfl.clipDP, args.lr)
        dp_raw_l = args.lr * sigma_dp
        if args.nfl.opt_target == 'val':
            l = dp_raw_l
        else:
            total_norm, _ = params_norm([torch.ones_like(p) for p in model.parameters()], norm_type=1)
            logger.info('model params by L1 norm={}'.format(total_norm))
            l = dp_raw_l * total_norm.item()
        u = 2*l
    return l, u


def dlg_inv_grad(args, ground_truth, labels, original_model, input_grad, tb_writer, cid, bid, train_epoch_round, input_type='raw'):    
    setup = inversefed_setup
    invgrad_config = dict(signed=True,
              boxed=True,
              cost_fn=args.nfl.cost_fn,  # sim
              indices='def',
              weights='equal',
              lr=args.nfl.dlg_lr,  # 0.1
              optim='adam',
              restarts=1,
              max_iterations=args.nfl.dlg_iter,  # 4000
              total_variation=args.nfl.tv_lambda,  # 1e-6
              init=args.nfl.dlg_img_init,  # randn / fname
              filter='none',
              lr_decay=True,
              scoring_choice='loss')

    setup = inversefed_setup
    
    # 针对CIFAR-10优化的参数配置
    if 'cifar' in args.dataset.lower():
        invgrad_config = dict(signed=True,
                  boxed=True,
                  cost_fn=args.nfl.cost_fn,  # sim
                  indices='def',
                  weights='equal',
                  lr=0.01,  # 降低学习率，CIFAR-10更复杂
                  optim='adam',
                  restarts=5,  # 增加重启次数
                  max_iterations=8000,  # 增加迭代次数
                  total_variation=1e-4,  # 增加总变分正则化
                  init='randn',  # 使用随机初始化
                  filter='none',
                  lr_decay=True,
                  scoring_choice='loss')
    else:
        # MNIST的原始配置
        invgrad_config = dict(signed=True,
                  boxed=True,
                  cost_fn=args.nfl.cost_fn,
                  indices='def',
                  weights='equal',
                  lr=args.nfl.dlg_lr,
                  optim='adam',
                  restarts=1,
                  max_iterations=args.nfl.dlg_iter,
                  total_variation=args.nfl.tv_lambda,
                  init=args.nfl.dlg_img_init,
                  filter='none',
                  lr_decay=True,
                  scoring_choice='loss')

    dm = torch.as_tensor(eval(f'inversefed.consts.{args.dataset}_mean'), **setup)[:, None, None]
    ds = torch.as_tensor(eval(f'inversefed.consts.{args.dataset}_std'), **setup)[:, None, None]
    # img_shape = (1, 28, 28) if 'mnist' in args.dataset else (3, 32, 32)
    if 'mnist' in args.dataset:
        img_shape = (1, 28, 28)
    elif 'tiny' in args.dataset:
        img_shape = (3, 64, 64)
    else:
        img_shape = (3, 32, 32)
    
    if input_type == 'updates':
        rec_machine = inversefed.FedAvgReconstructor(original_model, (dm, ds), args.local_batch_iter, args.lr, invgrad_config,
                                                     use_updates=True, num_images=args.batch_size)
    else:
        rec_machine = inversefed.GradientReconstructor(original_model, (dm, ds), invgrad_config, num_images=args.batch_size)
    
    try:
        output, stats = rec_machine.reconstruct(input_grad, labels, img_shape=img_shape, tb_writer=tb_writer, cid=cid, bid=bid)
        
        # 检查重建是否成功
        if 'opt' in stats and stats['opt'] == float('inf'):
            print("Warning: Reconstruction failed, results may not be reliable.")
        
    except Exception as e:
        print(f"Reconstruction failed with error: {e}")
        # 返回默认结果或跳过这次重建
        return None

    # 计算评估指标（添加错误处理）
    try:
        test_mse = (output.detach() - ground_truth).pow(2).mean().item()
        feat_mse = (original_model(output.detach())- original_model(ground_truth)).pow(2).mean().item()
        test_psnr = float(inversefed.metrics.psnr(output, ground_truth, factor=1/ds))
        test_ssim = float(np.mean([calculate_ssim(torch.unsqueeze(x, 0).cpu(), torch.unsqueeze(x_rec, 0).cpu()).cpu().detach().item() 
                            for x,x_rec in zip(ground_truth, output)]))
    except Exception as e:
        print(f"Error computing metrics: {e}")
        test_mse = float('inf')
        feat_mse = float('inf') 
        test_psnr = 0.0
        test_ssim = 0.0

    output = output.clone().detach()
    output.mul_(ds).add_(dm).clamp_(0, 1)
    output = output.cpu()
    
    gt_img = ground_truth.detach().mul_(ds).add_(dm).clamp_(0, 1).cpu()
    
    dlg_result_dict = dict(
        test_mse=test_mse, feat_mse=feat_mse, test_psnr=test_psnr, test_ssim=test_ssim,
        rec_img=output.numpy(), gt=gt_img.numpy()
    )
    
    pk.dump(dlg_result_dict, open(os.path.join(args.checkpoint_dir, f'dlg_result_E{train_epoch_round}.pkl'), 'wb'))
    
    tb_writer.add_scalar(f'C{cid}/dlg_B{bid}_mse', test_mse, train_epoch_round)
    tb_writer.add_scalar(f'C{cid}/dlg_B{bid}_feat_mse', feat_mse, train_epoch_round)
    tb_writer.add_scalar(f'C{cid}/dlg_B{bid}_psnr', test_psnr, train_epoch_round)
    tb_writer.add_scalar(f'C{cid}/dlg_B{bid}_ssim', test_ssim, train_epoch_round)
    tb_writer.add_images(f'C{cid}/dlg_B{bid}_imgs', output, train_epoch_round)
    tb_writer.add_images(f'C{cid}/dlg_B{bid}_imgs_raw', gt_img, train_epoch_round)


def fed_train():
    # ===0.0 config
    logger.info("#" * 100)
    logger.info(str(args))
    set_seed(args.seed)
    # ===0.1 init tb writer
    tb_writer = SummaryWriter(args.checkpoint_dir)

    # ===1.create clients and server
    clients, server = fed_init(args, tb_writer, args.dataset, args.shuffle)
    logger.info([(k,p.detach().norm().cpu().item()) for k, p in clients[0].model.named_parameters()])
    
    # args.nfl.apply_distortion = 'nfl'
    if args.nfl.apply_distortion == 'nfl':

        # 得到失真程度Δ，参考伪代码算法2 和公式2
        l, u = get_nfl_bounds(args, clients[0].model)

    # elif args.nfl.apply_distortion == 'barre' or args.nfl.apply_distortion == 'rpf':
    #     eps = args.nfl.eps  # ε ∈ [0.1, 0.9]
    #     # 线性映射，下界从 850→550，gap=50
    #     l = 887.5 - 375.0 * eps
    #     u = l + 50.0
    elif args.nfl.apply_distortion == 'barre' or args.nfl.apply_distortion == 'rpf':
        # eps = args.nfl.eps  # ε ∈ [0.1, 0.9]
        # # 线性映射，下界从 850→550，gap=50
        # # 50-15.0 * eps, + 25.0 eps=0.1-> psnr 8.5
        # l = 100 - 30.0 * eps
        # u = l + 25.0
        eps = args.nfl.eps
        alpha = 0.5
        l = 500 * (1.0 - eps)**alpha
        l = round(l.real, 2)
        u = l + 22.0
    else:
        l, u = -1, -1
    args.nfl.l, args.nfl.u = l, u  # 根据ε得到扰动的上下界。

        

    tb_writer.add_text('config', str(args.nfl))


    # ===2. FL process
    best_val_accs = [0.] * args.n_clients
    test_accs = [0.] * args.n_clients
    best_rounds = [-1] * args.n_clients
    client_loss_meter_list = [AvgMeter() for _ in range(args.n_clients)]
    comm_R = 0

    for train_epoch_round in range(args.global_epoch):
        # ===2.1 init for one global epoch
        logger.info("** Training Epoch {}, Communication Round:{} Start! **".format(train_epoch_round, comm_R))
        train_loader_list = list()
        for i in range(args.n_clients):
            train_loader_list.append(clients[i].trainloader)
            client_loss_meter_list[i].reset()

        # ===2.2 start one global epoch (n_batch)
        best_val_acc_global, best_te_acc_global = 0,0

        for batch_idx, clients_batch_data in enumerate(zip(*train_loader_list)):# 开始客户端训练
            # ===2.2.1: client sequential train
            for client_idx, c_batch_data in enumerate(clients_batch_data):  # 每个客户端的批次数据
                # === before
                if args.nfl.apply_dlg:  # 如果是DLG攻击,在“预热”或“正式”训练前，准备一个可供 DLG 攻击使用的模型副本，从而在实验中对比有/无隐私保护措施时，攻击者究竟能否通过梯度反演拿到原始数据。
                    clients[client_idx].frozen_net(False)
                    original_model = clients[client_idx].get_copied_model()
                    original_model.train()
                    

                # === local train @ client idx
                x, y = c_batch_data[0].to(device), c_batch_data[1].to(device) # x是4维的tensor，（batch，h，w，channels），y是1维的tensor

                if train_epoch_round <= args.nfl.warm_up_rounds - 1:  # 先进行预热阶段，预热几个轮次在进行隐私保护
                    loss_list, acc_list, grad_list, noisy_grad_list = clients[client_idx]. \
                        perform_nfl_train(x, y, comm_R, l, u, warming_up=True,
                                          u_loss_type=args.nfl.u_loss_type,
                                          nfl_lba=args.nfl.lba, clipDP=args.nfl.clipDP,
                                          privacy_measure=args.nfl.privacy, optimized_target=args.nfl.opt_target)
                else:    # 正式训练阶段。添加BARRE算法。          
                    if 'dp' in args.nfl.apply_distortion:
                        mecha = args.nfl.apply_distortion[3:]
                        loss_list, acc_list, grad_list, noisy_grad_list = clients[client_idx].\
                            perform_dp_train(x, y, comm_R,
                                            clip=args.nfl.clipDP, mechanism=mecha,
                                            eps=args.nfl.eps, clip_level=args.nfl.clipL,
                                            element_wise_rand=args.nfl.element_wise_rand)
                    elif args.nfl.apply_distortion == 'barre':
                        # 使用BARRE算法训练模型
                        loss_list, acc_list, grad_list, noisy_grad_list = clients[client_idx].\
                            perform_barre_train(x, y, comm_R, l=l, u=u,
                                            M=args.nfl.barre_M, clipDP=args.nfl.clipDP,noise_type=args.nfl.barre_noise_type,
                                            k_noise=args.nfl.barre_k_noise, alpha_noise=args.nfl.barre_alpha_noise,
                                            tau=args.nfl.barre_tau)
                    elif args.nfl.apply_distortion == 'nfl':  # nfl算法
                        loss_list, acc_list, grad_list, noisy_grad_list = clients[client_idx]. \
                            perform_nfl_train(x, y, comm_R, l, u, warming_up=False,
                                            u_loss_type=args.nfl.u_loss_type,
                                            nfl_lba=args.nfl.lba, clipDP=args.nfl.clipDP,
                                            privacy_measure=args.nfl.privacy, optimized_target=args.nfl.opt_target,
                                            element_wise_rand=args.nfl.element_wise_rand,
                                            dp_upratio=args.nfl.dp_upratio)
                    else:  #RPF算法
                        loss_list, acc_list, grad_list = clients[client_idx]. \
                            perform_rpf_train(x, y, comm_R, l, u, noise_type=args.noise_type,
                                            u_loss_type=args.nfl.u_loss_type,
                                            clipDP=args.nfl.clipDP,
                                            privacy_measure=args.nfl.privacy, optimized_target=args.nfl.opt_target)
                        noisy_grad_list = []


                for loss, acc in zip(loss_list, acc_list):
                    client_loss_meter_list[client_idx].update(loss)
                # record best val
                if batch_idx % 5 == 0:
                    val_acc = clients[client_idx].local_val()
                    tb_writer.add_scalar(f'C{client_idx}/val_acc', val_acc, comm_R)
                    if val_acc > best_val_accs[client_idx]:
                        best_val_accs[client_idx] = val_acc
                        test_accs[client_idx] = clients[client_idx].local_test()
                        best_rounds[client_idx] = train_epoch_round

                # === after (挑选某些batch的图片进行DLG攻击)
                if args.nfl.apply_dlg and batch_idx == 0 and client_idx == 0 and train_epoch_round in args.nfl.dlg_attack_epochs:
                    if args.nfl.dlg_know_grad == 'raw':
                        equiv_raw_grad = grad_list[0]
                    elif args.nfl.dlg_know_grad == 'noisy':
                        equiv_raw_grad = noisy_grad_list[0] if noisy_grad_list else grad_list[0]
                    elif args.nfl.dlg_know_grad == 'equiv':
                        updated_model = clients[client_idx].get_copied_model()
                        equiv_raw_grad = list((om.detach().clone() - tm.detach().clone()) / args.lr for om, tm in
                                    zip(original_model.parameters(), updated_model.parameters()))
                    elif args.nfl.dlg_know_grad == 'updates':
                        updated_model = clients[client_idx].get_copied_model()
                        equiv_raw_grad = list((um.detach().clone() - om.detach().clone()) for om, um in
                                              zip(original_model.parameters(), updated_model.parameters()))
                    else:
                        raise NotImplementedError('{} not valid'.format(args.nfl.dlg_know_grad))

                    dlg_inv_grad(args, x, y, original_model, equiv_raw_grad,  # 使用真实数据标签，原始模型，攻击者获取的梯度进行攻击。
                                 tb_writer, client_idx, batch_idx, train_epoch_round,
                                 args.nfl.dlg_know_grad)

                    
                    # 保存dlg的现场
                    dlg_save_dict = dict(
                        x = x, y = y,
                        ori_model = original_model.state_dict(),
                        equiv_raw_grad = equiv_raw_grad,
                        local_model_lr=args.lr,
                        cid=client_idx, batch_idx=batch_idx,
                        round=comm_R,
                        gE=train_epoch_round,
                    )
                    save_path = os.path.join(args.checkpoint_dir, 'dlg')
                    if not os.path.exists(save_path):
                        os.makedirs(save_path)
                    torch.save(dlg_save_dict, os.path.join(save_path, f'C{client_idx}_E{train_epoch_round}_B{batch_idx}.pkl'))
                    
            # ===2.2.2 server aggregation
            server.receive()
            server.send()
            comm_R += 1

        val_acc_global, te_acc_global = server.eval_global('val'), server.eval_global('test')
        tb_writer.add_scalar(f'global/val_acc', val_acc_global, train_epoch_round)
        tb_writer.add_scalar(f'global/test_acc', te_acc_global, train_epoch_round)

        # 将每轮的验证和测试准确率记录到日志文件
        metrics_file_path = os.path.join(args.checkpoint_dir, 'all_metrics_log.txt')
        with open(metrics_file_path, 'a') as metrics_file:
            metrics_file.write(f"{train_epoch_round},{val_acc_global},{te_acc_global}\n")


        if val_acc_global > best_val_acc_global:
            best_val_acc_global, best_te_acc_global = val_acc_global, te_acc_global
            with open(os.path.join(args.checkpoint_dir, 'best_metric.txt'), 'w') as f:
                f.writelines(['{},{},{}'.format(train_epoch_round, best_val_acc_global, best_te_acc_global)])
            torch.save(server.global_net, os.path.join(args.checkpoint_dir, 'server_best_global.pkl'))

        # ===2.3 end one global epoch
        # ===2.3.1 check early stop
        early_stop = True
        for i in range(args.n_clients):
            if train_epoch_round <= best_rounds[i] + args.early_stop_rounds:
                early_stop = False 
                break
        if early_stop:
            logger.info("early stoped at epoch{}, communication round:{}".format(train_epoch_round, comm_R))
            break
        # ===2.3.2 report metric for each global epoch
        logger.info("train epoch round:{}, communication round:{}".format(train_epoch_round, comm_R))
        for i in range(args.n_clients):
            loss_avg = client_loss_meter_list[i].get()
            logger.info("client:%2d, test acc:%2.6f, best epoch:%2d, loss:%2.6f" % (i, test_accs[i], best_rounds[i], loss_avg))

    # ===3. End of FL
    logger.info("** Federated Learning Finish! **")
    for i in range(args.n_clients):
        logger.info("client:%2d, test acc:%2.6f, best epoch:%2d" % (i, test_accs[i], best_rounds[i]))


if __name__ == "__main__":
    try:
        fed_train()
    except Exception as e:
        msg = traceback.format_exc()
        logger.exception(msg)
