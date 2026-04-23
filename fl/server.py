import numpy as np
import torch
import torch.nn as nn
import os
from basic.config import args, logger, device
from basic.models import LeNetZhuMNIST, Lenet5
from basic.utils import set_seed
import inversefed

# todo: only suitable for Fashion-MNIST
class Server:

    def __init__(self, clients, ds_name, arch, ckpt_dir,use_rp=False, rp_ratio=0.25, rp_eps = 0.2):
        self.clients = clients
        self.checkpoint_dir = ckpt_dir
        self.weights = []
        for client in clients:
            self.weights.append(client.train_size)
        self.weights = np.array(self.weights) / np.sum(self.weights)
        logger.info("client weights: %s" % str(self.weights.tolist()))
        
        
        # 参数应该放到函数上面
        self.use_rp = use_rp
        self.rp_ratio = rp_ratio
        self.rp_eps = rp_eps

        self.init_net(ds_name, arch)


    def frozen_net(self, frozen):
        for param in self.global_net.parameters():
            param.requires_grad = not frozen
        if frozen:
            self.global_net.eval()
        else:
            self.global_net.train()

    def init_net(self, ds_name, arch):
        set_seed(args.seed)
        self.global_net = nn.ModuleDict()


        num_channels=1 if 'mnist' in ds_name else 3
        num_classes_map = {'mnist': 10, 'fmnist': 10, 'emnist': 47, 'cifar10': 10, 'cifar100': 100, 'tiny': 200}
        num_classes = num_classes_map.get(ds_name, 200)
        model, _ = inversefed.construct_model(arch, num_classes=num_classes, num_channels=num_channels,use_rp=self.use_rp,rp_ratio=self.rp_ratio,rp_eps=self.rp_eps)
        self.global_net = model
        self.frozen_net(True)

        self.global_net.to(device)

        self.KL_criterion = nn.KLDivLoss(reduction="batchmean").to(device)
        self.CE_criterion = nn.CrossEntropyLoss().to(device)
        
    def receive(self):
        avg_param = {}
        params = []

        for client in self.clients:
            params.append(client.model.state_dict())
        for key in params[0].keys():
            avg_param[key] = params[0][key] * self.weights[0]
            for idx in range(1, len(self.clients)):
                avg_param[key] += params[idx][key] * self.weights[idx]
        self.global_net.load_state_dict(avg_param)

    def send(self):
        global_param = self.global_net.state_dict()
        for client in self.clients:
            client.model.load_state_dict(global_param)
    
    def eval_global(self, data_type='test'):
        assert data_type in ['val', 'test']
        corr_global, total_global = 0, 0
        for c in self.clients:
            corr, total = c.local_val(True) if data_type == 'val' else c.local_test(True)
            corr_global += corr
            total_global += total
        return corr_global / total_global