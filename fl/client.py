import copy
import math
from pyexpat import model
from basic.config import device
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from basic.config import logger, device
from basic.models import LeNetZhuMNIST, Lenet5
import numpy as np
import inversefed
# from tensorflow_privacy.privacy.analysis.compute_noise_from_budget_lib import compute_noise


# def add_delta(model, device="cpu"):
#     with torch.no_grad():
#         # modified 原来那样加是不会改变模型的，得这样
#         state_dict = model.state_dict()
#         for name in state_dict:
#             param = state_dict[name]
#             delta = torch.normal(torch.zeros(param.data.size()), 10).to(device)
#             # delta = torch.zeros_like(param.data, requires_grad=True).to(device)
#             delta.requires_grad = True
#             state_dict[name] = param + delta
#     return state_dict


# 这是根据aij文章的公式2，根据ε求得Δ，及失真程度。
def get_delta_norm_by_eps(eps, D=1, ca=1, c0=1, I=1600, p=0.5):
    """
    reverse Impl of equation(19)
    """
    I_term = np.power(I, p-1)
    delta_norm = (4 * D * (1-eps))/ca - (c0 * I_term)
    return delta_norm



# 这是公式2，根据Δ，及失真程度求得ε
def get_eps_by_delta_norm(delta_net, D=1, ca=1, c0=1, I=1600, p=0.5):
    """
    Impl of equation(19)
    """
    I_term = np.power(I, p-1)
    delta_norm = torch.norm(torch.stack([torch.norm(p, 2.0).to(device) for p in delta_net.parameters()]), 2.0)
    epsilon_p = 1 - ((ca * delta_norm + ca * c0 * I_term) / (4 * D))
    return delta_norm, epsilon_p



def params_norm(params, norm_type: float = 2.0, error_if_nonfinite: bool = False) -> torch.Tensor:
    norm_type = float(norm_type)
    if len(params) == 0:
        return torch.tensor(0.)
    device = params[0].device
    if norm_type == np.inf:
        norms = [p.detach().abs().max().to(device) for p in params]
        total_norm = norms[0] if len(norms) == 1 else torch.max(torch.stack(norms))
    else:
        all_param_norm = torch.stack([torch.norm(p.detach(), norm_type).to(device) for p in params])
        total_norm = torch.norm(all_param_norm, norm_type)
    if error_if_nonfinite and torch.logical_or(total_norm.isnan(), total_norm.isinf()):
        raise RuntimeError(
            f'The total norm of order {norm_type} for gradients from '
            '`parameters` is non-finite, so it cannot be clipped. To disable '
            'this error and scale the gradients by the non-finite norm anyway, '
            'set `error_if_nonfinite=False`')
    return total_norm, all_param_norm


def proj_by_norm_(parameters, min_norm, max_norm, norm_type=2):
    # 1. calc grad norm
    total_norm, _ = params_norm(parameters, norm_type=norm_type)

    # 2. calc norm-based scaling factor
    min_norm = torch.Tensor([min_norm]).to(total_norm.device)
    max_norm = torch.Tensor([max_norm]).to(total_norm.device)
    if total_norm < min_norm:
        coef = min_norm / (total_norm + 1e-9)
        # for p in parameters:
        #     p.detach().mul_(0.0).add_(min_norm/len())
    else:
        coef = torch.clamp(max_norm / total_norm, max=1.0)

    # 3. apply grad scaling
    for p in parameters:
        p.detach().mul_(coef.to(p.device))

    return total_norm, coef


def distortion_init(tb_writer, comm_R, client_id, model, l, num_distort_iter,
                    privacy_measure='nfl', optimized_target='sigma', element_wise_rand=True):
    """
    initializing the target distortion variable, return as ori_delta_state_dict
    privacy_measure: [nfl, dpl, dpn]
    optimized_target: [variance, value]
    method_type:
        nfl-fix: nfl + value, delta ~ Lap(l_k), scaling delta_norm => l_k
        nfl-learn: nfl + value, delta ~ Lap(l_k), scaling delta_norm => l_k
        dpl-nfl-value: dp + value, delta ~ Lap(l_k=sigma_dp)
        dpl-nfl-variance: dp + variance, delta=l_k=sigma_dp
    """
    assert privacy_measure in ['dp', 'nfl']
    assert optimized_target in ['val', 'sigma']
    ori_delta_state_dict = copy.deepcopy(model.state_dict())

    # 生成拉普拉斯分布的随机噪声，分布参数由 𝑙 控制：
    m = torch.distributions.laplace.Laplace(torch.tensor(0.0), torch.tensor(l*1.0))

    #1：L1 范数，适用于 dp。
    #2：L2 范数，适用于 nfl。
    norm_type = 1 if privacy_measure == 'dp' else 2

    # 1. initializing delta，对模型的所有参数初始化Δ。
    for k, v in ori_delta_state_dict.items():
        if '.bn' not in k:  # bn层的参数不需要初始化
            # 确保张量为浮点类型
            if not v.dtype.is_floating_point:
                ori_delta_state_dict[k] = v.float()
            
            # 如果 element_wise_rand=True，对每个元素生成独立噪声。
            # 如果 element_wise_rand=False，对整个参数生成相同的噪声。
            
            if element_wise_rand:
                delta = m.sample(ori_delta_state_dict[k].shape).to(ori_delta_state_dict[k].device)
            else:
                delta = m.sample([1]).to(ori_delta_state_dict[k].device)
            # 生成的delta作为噪声。参数先填充了1
            ori_delta_state_dict[k].data.fill_(1.0).mul_(delta)

    # 调用 params_norm 函数，计算模型所有非 BN 参数的初始范数
    init_norm, _ = params_norm([v for k,v in ori_delta_state_dict.items() if '.bn' not in k], norm_type=norm_type)
    tb_writer.add_scalar(f'C{client_id}/nfl_delta_norm_init', init_norm, comm_R * num_distort_iter)

    # 2. scaling delta norm (except for nfl-dp-val)
    # 若初始范数不等于目标范数 𝑙，则对扰动 Δ 进行缩放：
    if not (privacy_measure=='dp' and optimized_target=='val') and init_norm != l:
        coef = l / (init_norm + 1e-9)
        for k in ori_delta_state_dict:
            if '.bn' not in k:
                ori_delta_state_dict[k].data.mul_(coef)
        # 缩放后的范数再次计算
        init_norm_scaled, _ = params_norm([v for k,v in ori_delta_state_dict.items() if '.bn' not in k], norm_type=norm_type)
        tb_writer.add_scalar(f'C{client_id}/nfl_delta_norm_init_scaled', init_norm_scaled, comm_R * num_distort_iter)
    return ori_delta_state_dict, init_norm


def distortion_learning(tb_writer, comm_R, client_id, batch_data, batch_label, model, CE_criterion, u_loss_type='gap', raw_loss_val=0,
                        num_distort_iter=1, zeta=0.001, lba=10, u=12.0, l=0.,
                        privacy_measure='nfl', optimized_target='val', element_wise_rand=True, dp_upratio=2):
    # 首先初始化一个失真变量。
    ori_delta_state_dict, init_norm = distortion_init(tb_writer, comm_R, client_id, model, l, num_distort_iter,
                                                      privacy_measure, optimized_target, element_wise_rand)    
    norm_type = 1 if optimized_target == 'sigma' else 2
    
    if comm_R % 499 == 0:
        delta_stat = {}
        for k,v in ori_delta_state_dict.items():
            # 确保张量为浮点类型以进行统计计算
            v_float = v.detach().float() if not v.dtype.is_floating_point else v.detach()
            layer_size = torch.ones_like(v_float).sum().item()
            m, std, norm = v_float.mean().item(), v_float.std().item(), torch.norm(v_float, norm_type).item()
            delta_stat[k] = dict(mean=m, std=std,norm_avg=norm/layer_size)
            for kk,vv in delta_stat[k].items():
                tb_writer.add_scalar(f'{client_id}_{kk}/{k}_before', vv, comm_R * num_distort_iter)
            tb_writer.add_histogram(f'{client_id}_hist/{k}_before', v_float.cpu().numpy(), comm_R * num_distort_iter, bins=20)
        

    if privacy_measure == 'dp' and optimized_target == 'val':
        l, u = init_norm, dp_upratio*init_norm  # reevised from sigma --> L1/sum of sampled value
   
    for iter in range(num_distort_iter):
        # 1. utility loss (by combining net)
        delta_net = copy.deepcopy(model)
        delta_optim = optim.SGD(delta_net.parameters(), lr=zeta)
        delta_state_dict = delta_net.state_dict()
        
        # 确保delta_net的所有参数都是浮点类型
        for name in delta_state_dict:
            if not delta_state_dict[name].dtype.is_floating_point:
                delta_state_dict[name] = delta_state_dict[name].float()
        
        for name, delta_name in zip(delta_state_dict, ori_delta_state_dict):
            if '.bn' not in name:
                delta = ori_delta_state_dict[delta_name]
                if optimized_target == 'sigma':  # re-parameterization trick of laplacian
                    delta = torch.abs(delta) * laplace_noise(delta.shape, 1, delta.device)
                
                # 确保数据类型匹配
                if delta_state_dict[name].dtype != delta.dtype:
                    delta = delta.to(delta_state_dict[name].dtype)
                
                delta_state_dict[name] += delta
        
        delta_net.load_state_dict(delta_state_dict)

        # 计算效用损失
        pred = delta_net(batch_data)
        if u_loss_type == 'gap':
            utility_loss = torch.square(CE_criterion(pred, batch_label) - raw_loss_val)
        else:
            utility_loss = CE_criterion(pred, batch_label)
        loss = utility_loss

        # 2. privacy loss
        if lba != 0:
            for key, param in delta_net.named_parameters():
                if '.bn' not in key:
                    param.data = ori_delta_state_dict[key].data
            delta_norm = torch.norm(torch.stack([torch.norm(p, norm_type) for p in delta_net.parameters()]), norm_type)
            dummy_privacy_budget = -delta_norm  # simplified
            
            loss += float(lba) * dummy_privacy_budget
            tb_writer.add_scalar(f'C{client_id}/nfl_delta_norm', delta_norm.item(), comm_R * num_distort_iter + iter)

        # 3.1 update delta
        loss.backward()
        nfl_grad_norm, _ = params_norm([p.grad for p in delta_net.parameters()], norm_type=2)
        delta_optim.step()

        # 3.2 delta norm projection
        norm_type = 2 if privacy_measure == 'nfl' else 1
        total_norm_old, coef = proj_by_norm_(list(delta_net.parameters()), l, u, norm_type=norm_type)
        delta_norm_clipped, _ = params_norm(list(delta_net.parameters()), norm_type=norm_type)
        ori_delta_state_dict = delta_net.state_dict()
        
        # 4. record metrics
        if lba !=0:
            tb_writer.add_scalar(f'C{client_id}/nfl_delta_norm', delta_norm.item(), comm_R * num_distort_iter + iter)
        tb_writer.add_scalar(f'C{client_id}/nfl_grad_norm', nfl_grad_norm, comm_R * num_distort_iter + iter)
        tb_writer.add_scalar(f'C{client_id}/nfl_delta_norm_clipped', delta_norm_clipped, comm_R * num_distort_iter + iter)
        tb_writer.add_scalar(f'C{client_id}/nfl_u_loss', utility_loss.item(), comm_R * num_distort_iter + iter)
        tb_writer.add_scalar(f'C{client_id}/nfl_total_loss', loss.item(), comm_R * num_distort_iter + iter)

    if comm_R % 499 == 0:
        for k,v in ori_delta_state_dict.items():
            # 确保张量为浮点类型以进行统计计算
            v_float = v.detach().float() if not v.dtype.is_floating_point else v.detach()
            layer_size = torch.ones_like(v_float).sum().item()
            m, std, norm = v_float.mean().item(), v_float.std().item(), torch.norm(v_float, norm_type).item()
            stat_after = dict(mean=m, std=std,norm_avg=norm/layer_size)
            for name, val in stat_after.items():
                tb_writer.add_scalar(f'{client_id}_{name}/{k}_after', val, comm_R * num_distort_iter)
                tb_writer.add_scalar(f'{client_id}_{name}/{k}_gap', val-delta_stat[k][name], comm_R * num_distort_iter)
            tb_writer.add_histogram(f'{client_id}_hist/{k}_after', v_float.cpu().numpy(), comm_R * num_distort_iter, bins=20)

    return ori_delta_state_dict


def gaussian_noise(data_shape, sigma, device=None):
    return torch.normal(0, sigma, data_shape).to(device)


def laplace_noise(data_shape, scale, device=None):
    m = torch.distributions.laplace.Laplace(torch.tensor(0.0), torch.tensor(scale))
    return m.sample(data_shape).to(device)


def dp_scale_laplace(eps, clip, lr):
    sens = 2 * clip * lr
    scale = sens / eps
    return scale


class Client:

    def __init__(self, client_id, ds_name, arch, trainset, valset, testset, shuffle=False, apply_distortion=True, distortion_iter=5,
                 local_batch_iter=1, model_optim="adam", zeta=0.05, lr=3e-4, bs=8, wd=0, le=10, device="cpu",
                 tb_writer=None,use_rp=False, rp_ratio=0.25, rp_eps = 0.2):
        super(Client, self).__init__()

        assert tb_writer is not None
        self.tb_writer = tb_writer

        self.id = client_id
        self.device = device
        self.apply_distortion = apply_distortion
        self.model_optim = model_optim.lower()

        self.trainset = trainset
        pin_memory = False
        self.trainloader = DataLoader(trainset, batch_size=bs, shuffle=shuffle, num_workers=0, pin_memory=pin_memory)
        self.valloader = DataLoader(valset, batch_size=bs * 50, shuffle=False, num_workers=0, pin_memory=pin_memory)
        self.testloader = DataLoader(testset, batch_size=bs * 50, shuffle=False, num_workers=0, pin_memory=pin_memory)
        self.train_size = len(trainset)
        self.val_size = len(valset)
        self.test_size = len(testset)

        self.zeta = zeta

        self.local_lr = lr
        self.weight_decay = wd
        self.local_epoch = le
        self.model_optimizer = None

        self.local_batch_iter = local_batch_iter
        self.distortion_iter = distortion_iter

        logger.info("client:%2d, train_size:%4d, val_size:%4d, test_size:%4d" % (
            self.id, self.train_size, self.val_size, self.test_size))
        logger.info("local_batch_iter:%2d, distortion_iter:%2d." % (self.local_batch_iter, self.distortion_iter))

        self.CE_criterion = nn.CrossEntropyLoss().to(device)
        self.MSE_criterion = nn.MSELoss().to(device)
        self.model = None

        self.accum_grad_list = list()
        
        # RPF参数
        self.use_rp = use_rp
        self.rp_ratio = rp_ratio
        self.rp_eps = rp_eps

        self.init_net(ds_name, arch)

    def get_copied_model(self):
        return copy.deepcopy(self.model)

    def frozen_net(self, frozen):
        for param in self.model.parameters():
            param.requires_grad = not frozen
        if frozen:
            self.model.eval()
        else:
            self.model.train()

    def init_net(self, ds_name, arch):
        """frozen all models' parameters, unfrozen when need to train"""
        num_channels=1 if 'mnist' in ds_name else 3
        num_classes_map = {'mnist': 10, 'fmnist': 10, 'emnist': 47, 'cifar10': 10, 'cifar100': 100, 'tiny': 200}
        num_classes = num_classes_map.get(ds_name, 200)
        model, _ = inversefed.construct_model(arch, num_classes=num_classes, num_channels=num_channels ,use_rp=self.use_rp, rp_ratio=self.rp_ratio, rp_eps = self.rp_eps)

        self.model = model
        self.frozen_net(True)

        if self.model_optim == "adam":
            self.model_optimizer = optim.Adam(self.model.parameters(), lr=self.local_lr,
                                              weight_decay=self.weight_decay)
        elif self.model_optim == "sgd":
            self.model_optimizer = optim.SGD(self.model.parameters(), lr=self.local_lr)
        else:
            raise Exception("Does not support {} optimizer for now.".format(self.model_optim))

        self.model.to(device)

    def local_test(self, return_count=False):
        self.frozen_net(True)
        correct, total = 0, 0
        with torch.no_grad():
            for batch, (x, y) in enumerate(self.testloader):
                x = x.to(device)
                y = y.to(device)
                pred = self.model(x)
                correct += torch.sum((torch.argmax(pred, dim=1) == y).float())
                total += x.size(0)
        return (correct / total).item() if not return_count else (correct, total)

    def local_val(self, return_count=False):
        self.frozen_net(True)
        correct, total = 0, 0
        with torch.no_grad():
            for batch, (x, y) in enumerate(self.valloader):
                x = x.to(device)
                y = y.to(device)
                pred = self.model(x)
                correct += torch.sum((torch.argmax(pred, dim=1) == y).float())
                total += x.size(0)
        return (correct / total).item() if not return_count else (correct, total)

    # def privacy_leakage(self, params):
    #     d_sum = 0.0
    #     for param in params:
    #         print("param:", param)
    #         # for v in param:
    #         d_sum += param.pow(2).sum()
    #
    #     d_norm = torch.sqrt(d_sum)
    #     print("d_norm", d_norm)
    #     return 1 - (d_norm + 0.1) / 2

    def perform_dp_train(self, x, y, comm_R, clip=12., mechanism='laplace', eps=5, clip_level='sample', element_wise_rand=True):
        assert clip_level in ['sample', 'batch']
        assert mechanism in ['laplace', 'gaussian']
        clip_norm_type = 1 if mechanism == 'laplace' else 2
        loss_val_list = list()
        acc_val_list = list()
        grad_list = []
        noisy_grad_list = []

        sample_wise_CE = nn.CrossEntropyLoss(reduction='none').to(device)

        clipped_grads = {name: torch.zeros_like(param) for name, param in self.model.named_parameters()}
        for i in range(self.local_batch_iter):
            self.frozen_net(False)
            self.model_optimizer.zero_grad()
            pred = self.model(x)
            loss_samples = sample_wise_CE(pred, y)
            loss_mean = loss_samples.mean()

            # sava raw grad
            client_grad = torch.autograd.grad(loss_mean, self.model.parameters(), retain_graph=True)
            original_grad = list((g.detach().clone() for g in client_grad))
            grad_list.append(original_grad)

            ## 1. clip
            if clip_level == 'sample':  # 单样本缩放
                grad_norms = torch.Tensor([0.0]).to(pred.device)
                for i in range(loss_samples.size()[0]):
                    loss_samples[i].backward(retain_graph=True)
                    grad_norms += torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                                                 max_norm=clip, norm_type=clip_norm_type)
                    for name, param in self.model.named_parameters():
                        clipped_grads[name] += param.grad
                    self.model.zero_grad()
                grad_norms /= loss_samples.size()[0]
            else:
                loss_mean.backward(retain_graph=True)
                grad_norms = torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                                            max_norm=clip, norm_type=clip_norm_type)
                for name, param in self.model.named_parameters():
                    clipped_grads[name] += param.grad
                self.model.zero_grad()

            self.tb_writer.add_scalar(f'C{self.id}/raw_grad_norm', grad_norms, comm_R)

            ## 2. add noise (gaussian or laplace)
            sens = 20 * clip * self.local_lr
            if mechanism == 'laplace':
                scale = sens / eps
            else:
                delta = 1e-5
                scale = sens * math.sqrt(2 * math.log(1.25 / delta)) / eps
            self.tb_writer.add_scalar(f'C{self.id}/dp_scale', scale, comm_R)
            for name, param in self.model.named_parameters():
                if mechanism == 'laplace':
                    if element_wise_rand:
                        noise = laplace_noise(clipped_grads[name].shape, scale, device=param.device)
                    else:
                        noise = laplace_noise([1], scale, device=param.device) * torch.ones_like(clipped_grads[name])
                else:
                    if element_wise_rand:
                        noise = gaussian_noise(clipped_grads[name].shape, scale, device=param.device)
                    else:
                        noise = gaussian_noise([1], scale, device=param.device) * torch.ones_like(clipped_grads[name])
                clipped_grads[name] += noise
                param.grad = clipped_grads[name]

            # 按照相同的顺序生成noisy_grad并保存，方便DLG使用
            step_noisy_grad = [p.grad.detach().clone() for p in self.model.parameters()]
            noisy_grad_list.append(step_noisy_grad)

            # update local model
            self.model_optimizer.step()

            loss_val = loss_samples.mean().item()
            loss_val_list.append(loss_val)
            acc = torch.sum((torch.argmax(pred, dim=1) == y).float()) / x.size(0)
            acc_val_list.append(acc.item())
            self.tb_writer.add_scalar(f'C{self.id}/train_loss', loss_val, comm_R)
            self.tb_writer.add_scalar(f'C{self.id}/train_acc', acc, comm_R)

            self.frozen_net(True)
        return loss_val_list, acc_val_list, grad_list, noisy_grad_list  # 返回损失值列表、准确率列表、原始梯度列表和噪声梯度列表



    def perform_rpf_train(self, x, y, comm_R, l, u,
                      noise_type=2, k_noise=5, alpha_noise=0.01,
                      clipDP=-1, u_loss_type='direct',
                      privacy_measure='nfl', optimized_target='val'):
        """
        改进的RPF训练方法，保留输入噪声注入逻辑，移除NFL失真学习与自适应权重。

        参数:
        - noise_type: 噪声类型 (0=无噪声, 1=双区间噪声, 2=优化对抗噪声)
        - k_noise: 优化噪声迭代次数
        - alpha_noise: 优化噪声学习率
        - clipDP: 梯度裁剪阈值，若 <=0 则不裁剪
        - u_loss_type: 上行隐私损失类型，可选 'gap' 或 'direct'
        - optimized_target: 优化目标，可选 'val' 或 'sigma'
        """
        assert optimized_target in ['val', 'sigma']
        if self.distortion_iter > 0:
            assert u_loss_type in ['gap', 'direct']

        loss_val_list = []
        acc_val_list = []
        grad_list = []

        for i in range(self.local_batch_iter):
            # 解冻网络并清零梯度
            self.frozen_net(False)
            self.model_optimizer.zero_grad()

            # 根据噪声类型注入输入噪声
            if noise_type == 1:
                x_input, noise_norm = self.add_bipolar_noise(x, l, u)
            elif noise_type == 2:
                x_input, noise_norm = self.add_op_noise(x, y, self.model,
                                                    l, u, opt_steps=k_noise, lr=alpha_noise)
            else:
                x_input = x
                noise_norm = 0.0

            # 前向计算与损失
            pred = self.model(x_input)
            loss = self.CE_criterion(pred, y)

            # 记录梯度
            grads = torch.autograd.grad(loss, self.model.parameters(), retain_graph=True)
            grad_list.append([g.detach().clone() for g in grads])

            # 反向传播
            loss.backward()

            # 梯度裁剪
            if clipDP > 0:
                raw_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=clipDP, norm_type=2)
                self.tb_writer.add_scalar(f'C{self.id}/raw_grad_norm', raw_norm,
                                        comm_R * self.local_batch_iter + i)

            # 参数更新
            self.model_optimizer.step()

            # 记录损失与准确率（基于原始输入预测准确率）
            loss_val_list.append(loss.item())
            acc = (self.model(x).argmax(dim=1) == y).float().mean().item()
            acc_val_list.append(acc)

            # TensorBoard 日志
            self.tb_writer.add_scalar(f'C{self.id}/train_loss', loss.item(),
                                    comm_R * self.local_batch_iter + i)
            self.tb_writer.add_scalar(f'C{self.id}/train_acc', acc,
                                    comm_R * self.local_batch_iter + i)
            if noise_type > 0:
                # 如果noise_norm是tensor，则取mean.item()，否则直接当作float
                if isinstance(noise_norm, float):
                    norm_val = noise_norm
                else:
                    norm_val = noise_norm.mean().item()
                self.tb_writer.add_scalar(f'C{self.id}/noise_norm', norm_val,
                                        comm_R * self.local_batch_iter + i)

            # 再次冻结网络
            self.frozen_net(True)

        return loss_val_list, acc_val_list, grad_list

    def perform_nfl_train(self, x, y, comm_R, l, u, warming_up=False, nfl_lba=10.,
                        clipDP=-1, u_loss_type='direct', privacy_measure='nfl', optimized_target='val',
                        element_wise_rand=True, dp_upratio=2):
        assert optimized_target in ['val', 'sigma']
        if self.distortion_iter > 0:
            assert u_loss_type in ['gap', 'direct']
        loss_val_list = list()
        acc_val_list = list()
        grad_list = []
        noisy_grad_list = []
        for i in range(self.local_batch_iter):
            self.frozen_net(False)

            self.model_optimizer.zero_grad()
            pred = self.model(x)
            loss = self.CE_criterion(pred, y)

            # sava raw grad
            client_grad = torch.autograd.grad(loss, self.model.parameters(), retain_graph=True)
            original_grad = list((g.detach().clone() for g in client_grad))
            grad_list.append(original_grad)
            loss.backward()

            if clipDP > 0:
                raw_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=clipDP, norm_type=1)
                self.tb_writer.add_scalar(f'C{self.id}/raw_grad_norm', raw_norm, comm_R)

            self.model_optimizer.step()

            loss_val = loss.item()
            loss_val_list.append(loss_val)
            acc = torch.sum((torch.argmax(pred, dim=1) == y).float()) / x.size(0)
            acc_val_list.append(acc.item())
            self.tb_writer.add_scalar(f'C{self.id}/train_loss', loss, comm_R)
            self.tb_writer.add_scalar(f'C{self.id}/train_acc', acc, comm_R)

            if not warming_up and self.apply_distortion == 'nfl':
                # ==1. calculate delta (distortion)
                ori_delta_state_dict = distortion_learning(self.tb_writer, comm_R, self.id, x, y, self.model,
                                                           self.CE_criterion,
                                                           u_loss_type=u_loss_type, raw_loss_val=loss_val,
                                                           num_distort_iter=self.distortion_iter, zeta=self.zeta,
                                                           lba=nfl_lba, u=u, l=l,
                                                           privacy_measure=privacy_measure, optimized_target=optimized_target,
                                                           element_wise_rand=element_wise_rand,
                                                           dp_upratio=dp_upratio)

                # ==2. add delta to the original model
                ori_model_state_dict = self.model.state_dict()
                for name, delta_name in zip(ori_delta_state_dict, ori_model_state_dict):
                    if '.bn' not in name:
                        delta = ori_delta_state_dict[delta_name]
                        if optimized_target == 'val':
                            delta_to_add = delta
                        else:
                            # use abs, because scale must be positive
                            delta_to_add = torch.distributions.laplace.Laplace(torch.zeros_like(delta), torch.abs(delta)).sample([1])[0]
                            delta_to_add.to(delta.device)
                        ori_model_state_dict[name] = ori_model_state_dict[name] + delta_to_add
                        # print("==> distorted:", d_model_state_dict[delta_name])
                self.model.load_state_dict(ori_model_state_dict)

                # ==3. compute grad to distorted model
                # save grad
                noisy_pred = self.model(x)
                noisy_loss = self.CE_criterion(noisy_pred, y)
                noisy_grad = torch.autograd.grad(noisy_loss, self.model.parameters(), retain_graph=True)
                _noisy_grad_ = list((g.detach().clone() for g in noisy_grad))
                noisy_grad_list.append(_noisy_grad_)
                self.model.zero_grad()

            self.frozen_net(True)
        return loss_val_list, acc_val_list, grad_list, noisy_grad_list

    def perform_barre_train(self, x, y, comm_R, l, u, M=3, clipDP=-1, noise_type=1, k_noise=5, alpha_noise=0.01, loss_weight=0.5, tau=1.0):
        """
        BARRE算法训练过程（对应论文Algorithm 2: ClientBatchGrad）

        对每个mini-batch训练M个学习器（链式继承），并基于验证集损失计算软权重 α，
        使用 softmax(-L_val / τ) 进行“混合上传”。
        由于本仓库采用 FedAvg 的参数聚合（而不是直接上传梯度），这里用 α 对 M 个学习器参数做线性混合，
        得到等价的“混合后本地模型”用于服务器聚合。

        参数:
        - x: 输入数据
        - y: 目标标签
        - comm_R: 通信轮次
        - l: 噪声范数下界
        - u: 噪声范数上界
        - M: 学习器数量
        - noise_type: 噪声类型 (0=不加噪声, 1=双区间噪声, 2=优化对抗噪声)
        - k_noise: 噪声优化迭代次数 (对应add_op_noise的opt_steps)
        - alpha_noise: 噪声优化学习率 (对应add_op_noise的lr)
        - loss_weight: 原始损失与噪声损失的权重平衡因子 (论文中的λ)
        - tau: 软权重温度参数 τ，越大越接近均匀混合

        返回:
        - loss_val_list, acc_val_list, grad_list, noisy_grad_list
        """
        loss_val_list = []
        acc_val_list = []
        grad_list = []
        noisy_grad_list = []

        model_list = []
        total_loss_list = []

        for m in range(M):
            if m == 0:
                train_model = copy.deepcopy(self.model)
            else:
                train_model = copy.deepcopy(model_list[m - 1])

            if self.model_optim == "adam":
                optimizer = optim.Adam(train_model.parameters(), lr=self.local_lr, weight_decay=self.weight_decay)
            elif self.model_optim == "sgd":
                optimizer = optim.SGD(train_model.parameters(), lr=self.local_lr)
            else:
                raise Exception("Does not support {} optimizer for now.".format(self.model_optim))

            for i in range(self.local_batch_iter):
                train_model.train()
                optimizer.zero_grad()

                x_noisy, noise_norm = self._apply_noise(x, y, train_model, l, u, noise_type, k_noise, alpha_noise)

                pred_original = train_model(x)
                loss_original = self.CE_criterion(pred_original, y)

                pred_noisy = train_model(x_noisy)
                loss_noisy = self.CE_criterion(pred_noisy, y)

                total_loss = loss_weight * loss_original + (1 - loss_weight) * loss_noisy

                if m == 0 and i == 0:
                    train_model.zero_grad()
                    client_grad = torch.autograd.grad(loss_original, train_model.parameters(), retain_graph=True)
                    original_grad = [g.detach().clone() for g in client_grad]
                    grad_list.append(original_grad)

                total_loss.backward()

                if clipDP > 0:
                    raw_norm = torch.nn.utils.clip_grad_norm_(train_model.parameters(), max_norm=clipDP, norm_type=2)
                    if m == 0:
                        self.tb_writer.add_scalar(f'C{self.id}/raw_grad_norm', raw_norm, comm_R * self.local_batch_iter + i)

                optimizer.step()

                loss_val = total_loss.item()
                acc = torch.sum((torch.argmax(pred_original, dim=1) == y).float()) / x.size(0)
                loss_val_list.append(loss_val)
                acc_val_list.append(acc.item())

                if m == 0:
                    self.tb_writer.add_scalar(f'C{self.id}/train_total_loss', loss_val, comm_R * self.local_batch_iter + i)
                    self.tb_writer.add_scalar(f'C{self.id}/train_acc', acc.item(), comm_R * self.local_batch_iter + i)

            val_total_loss = self.evaluate_model_loss(train_model, noise_type, l, u, k_noise, alpha_noise, loss_weight)
            total_loss_list.append(val_total_loss)
            self.tb_writer.add_scalar(f'C{self.id}/barre_classifier_{m}_val_total_loss', val_total_loss, comm_R)
            model_list.append(train_model)

        # ====== FedBARRE核心改动：由“选最优单模型”改为“软权重混合” ======
        val_losses_tensor = torch.tensor(total_loss_list, device=device, dtype=torch.float32)
        tau_safe = max(float(tau), 1e-6)
        alpha = torch.softmax(-val_losses_tensor / tau_safe, dim=0)
        alpha_cpu = alpha.detach().cpu().numpy()

        # 记录混合权重与熵（用于观察是否接近硬选择）
        for m, a_m in enumerate(alpha_cpu):
            self.tb_writer.add_scalar(f'C{self.id}/barre_alpha_{m}', float(a_m), comm_R)
        alpha_entropy = float(-(alpha * torch.log(alpha + 1e-12)).sum().item())
        self.tb_writer.add_scalar(f'C{self.id}/barre_alpha_entropy', alpha_entropy, comm_R)
        self.tb_writer.add_scalar(f'C{self.id}/barre_tau', tau_safe, comm_R)

        # 参数线性混合：theta_mix = sum_m alpha_m * theta_m
        mixed_state_dict = copy.deepcopy(model_list[0].state_dict())
        with torch.no_grad():
            for key in mixed_state_dict.keys():
                mixed_state_dict[key] = mixed_state_dict[key].detach().clone() * 0.0
                for m in range(M):
                    mixed_state_dict[key].add_(model_list[m].state_dict()[key] * alpha[m])
        self.model.load_state_dict(mixed_state_dict)

        # 计算加噪后的梯度（用于隐私分析）
        self.model.train()
        self.model.zero_grad()

        # raw grad from mixed model (更贴近攻击者真实可见上传)
        final_pred = self.model(x)
        final_loss = self.CE_criterion(final_pred, y)
        final_grad = torch.autograd.grad(final_loss, self.model.parameters(), retain_graph=True)
        grad_list = [[g.detach().clone() for g in final_grad]]

        x_final_noisy, _ = self._apply_noise(x, y, self.model, l, u, noise_type, k_noise, alpha_noise)
        noisy_pred = self.model(x_final_noisy)
        noisy_loss = self.CE_criterion(noisy_pred, y)
        noisy_grad = torch.autograd.grad(noisy_loss, self.model.parameters(), retain_graph=True)
        _noisy_grad_ = [g.detach().clone() for g in noisy_grad]
        noisy_grad_list.append(_noisy_grad_)

        self.model.eval()

        return loss_val_list, acc_val_list, grad_list, noisy_grad_list

    def _apply_noise(self, x, y, model, l, u, noise_type, k_noise, alpha_noise):
        """根据噪声类型对输入数据添加噪声并clamp到合法范围"""
        if noise_type == 0:
            return x.clone(), 0.0
        elif noise_type == 1:
            x_noisy, noise_norm = self.add_bipolar_noise(x, l, u)
        elif noise_type == 2:
            x_noisy, noise_norm = self.add_op_noise(x, y, model, l, u, opt_steps=k_noise, lr=alpha_noise)
        else:
            raise ValueError(f"Unknown barre_noise_type={noise_type}. Valid values: 0 (no noise), 1 (bipolar), 2 (optimized)")
        # Clamp到合法输入范围
        x_noisy = x_noisy.clamp(0.0, 1.0)
        return x_noisy, noise_norm

        
    def add_bipolar_noise(self, x, l, u):
        """
        添加双区间噪声到输入数据
        噪声范数在[l,u]范围内，方向随机为正负
        
        Args:
            x: 输入张量 (batch_size, ...)
            l: 噪声范数下界
            u: 噪声范数上界
        
        Returns:
            noisy_x: 添加噪声后的数据
            actual_norms: 实际噪声范数 (用于验证)
        """
        # 生成标准正态分布噪声
        noise = torch.randn_like(x)
        
        # 计算每个样本的噪声范数 (除了batch维度)
        noise_norm = torch.norm(noise, p=2, dim=tuple(range(1, noise.dim())), keepdim=True)
        
        # 随机采样目标范数在[l, u]区间
        batch_size = x.size(0)
        target_norms = torch.rand(batch_size, *([1] * (x.dim() - 1))).to(x.device) * (u - l) + l
        
        # 随机选择正负方向
        signs = torch.randint(0, 2, (batch_size, *([1] * (x.dim() - 1)))).float().to(x.device) * 2 - 1
        target_norms = target_norms * signs
        
        # print(f"Target norms range: [{l}, {u}]")
        
        # 标准化噪声并缩放到目标范数
        noise_normalized = noise / (noise_norm + 1e-8)
        final_noise = noise_normalized * target_norms
        
        # 验证实际范数
        actual_norms = torch.norm(final_noise, p=2, dim=tuple(range(1, final_noise.dim())), keepdim=True)
        
        # 记录统计信息
        mean_norm = actual_norms.mean().item()
        self.tb_writer.add_scalar(f'C{self.id}/bipolar_noise_norm_mean', mean_norm, 0)
        
        return x + final_noise, actual_norms.squeeze()



   

    def add_op_noise(self, x, y, model, l, u, opt_steps=5, lr=0.01):
        """
        添加双区间噪声后，通过梯度下降优化噪声，使其对模型性能的影响最小。
        目标：在隐私预算约束下最小化性能损失
        - x: 输入 batch
        - y: 真实标签
        - model: 用于评估的模型
        - l, u: 噪声范数下限与上限（隐私预算约束）
        - opt_steps: 梯度优化步数 (对应配置 barre_k_noise)
        - lr: 噪声梯度下降学习率 (对应配置 barre_alpha_noise)
        """
        was_training = model.training
        model.eval()

        # 冻结模型参数，防止梯度泄漏到模型上
        for param in model.parameters():
            param.requires_grad_(False)

        # 1. 生成基础双区间噪声
        batch_size = x.size(0)
        noise = torch.randn_like(x)
        noise_norm = torch.norm(noise, p=2, dim=tuple(range(1, noise.dim())), keepdim=True)

        # 随机选择区间：True表示上区间[l,u]，False表示下区间[0,l]
        use_upper_interval = torch.rand(batch_size, 1, 1, 1, device=x.device) > 0.1

        # 为每个样本分配目标范数
        upper_norms = torch.rand(batch_size, 1, 1, 1, device=x.device) * (u - l) + l  # [l, u]
        lower_norms = torch.rand(batch_size, 1, 1, 1, device=x.device) * l  # [0, l]
        target_norms = torch.where(use_upper_interval, upper_norms, lower_norms)

        # 初始化噪声到目标范数
        noise = noise / (noise_norm + 1e-8) * target_norms

        # 2. 对噪声进行优化以最小化性能影响
        noise.requires_grad_(True)
        optimizer = torch.optim.Adam([noise], lr=lr)

        for step in range(opt_steps):
            optimizer.zero_grad()
            x_noisy = x + noise
            preds = model(x_noisy)

            loss = self.CE_criterion(preds, y)
            loss.backward()

            optimizer.step()

            # 投影回双区间约束（向量化实现）
            with torch.no_grad():
                current_norms = torch.norm(noise, p=2, dim=tuple(range(1, noise.dim())), keepdim=True)
                scale = torch.ones_like(current_norms)

                # 上区间 [l, u] 的投影
                upper_too_small = use_upper_interval & (current_norms < l)
                upper_too_large = use_upper_interval & (current_norms > u)
                scale = torch.where(upper_too_small, l / (current_norms + 1e-8), scale)
                scale = torch.where(upper_too_large, u / (current_norms + 1e-8), scale)

                # 下区间 [0, l] 的投影
                lower_too_large = (~use_upper_interval) & (current_norms > l)
                scale = torch.where(lower_too_large, l / (current_norms + 1e-8), scale)

                noise.data.mul_(scale)

        noise.requires_grad_(False)

        # 3. 记录与返回
        final_norms = torch.norm(noise, p=2, dim=tuple(range(1, noise.dim())), keepdim=True)
        actual_norm = final_norms.mean().item()
        self.tb_writer.add_scalar(f'C{self.id}/_opt_norm', actual_norm, 0)

        # 恢复模型参数梯度状态和训练模式
        for param in model.parameters():
            param.requires_grad_(True)
        if was_training:
            model.train()

        return x + noise.detach(), actual_norm


    def evaluate_model_loss(self, model, noise_type, l, u, k_noise, alpha_noise, loss_weight, max_samples=200):
        """
        在验证集上评估模型的总体损失（原始损失 + 噪声损失的加权和）
        用于BARRE中选择最优learner，验证时统一使用廉价的双区间随机噪声来加速。

        参数:
        - model: 要评估的模型
        - noise_type: 噪声类型（此参数保留以兼容接口，验证时统一用bipolar噪声加速）
        - l: 噪声范数下界
        - u: 噪声范数上界
        - k_noise: 噪声优化迭代次数（验证时不使用）
        - alpha_noise: 噪声优化学习率（验证时不使用）
        - loss_weight: 原始损失与噪声损失的权重平衡因子
        - max_samples: 最大评估样本数

        返回:
        - total_loss: 加权总体损失
        """
        was_training = model.training
        model.eval()
        total_original_loss = 0.0
        total_noisy_loss = 0.0
        total_samples = 0

        with torch.no_grad():
            for batch_x, batch_y in self.valloader:
                batch_size = batch_y.size(0)

                if total_samples + batch_size > max_samples:
                    needed = max_samples - total_samples
                    batch_x = batch_x[:needed]
                    batch_y = batch_y[:needed]
                    batch_size = needed

                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)

                # 计算原始损失
                original_pred = model(batch_x)
                original_loss = self.CE_criterion(original_pred, batch_y)
                total_original_loss += original_loss.item() * batch_size

                # 计算噪声损失 — 统一使用bipolar噪声加速模型选择
                batch_x_noisy, _ = self.add_bipolar_noise(batch_x, l, u)
                noisy_pred = model(batch_x_noisy)
                noisy_loss = self.CE_criterion(noisy_pred, batch_y)
                total_noisy_loss += noisy_loss.item() * batch_size

                total_samples += batch_size
                if total_samples >= max_samples:
                    break

        avg_original_loss = total_original_loss / total_samples if total_samples > 0 else 0.0
        avg_noisy_loss = total_noisy_loss / total_samples if total_samples > 0 else 0.0
        total_loss = loss_weight * avg_original_loss + (1 - loss_weight) * avg_noisy_loss

        if was_training:
            model.train()
        return total_loss
