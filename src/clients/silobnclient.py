import os
import copy
import torch
import random
import numpy as np

from torch.utils import data
from collections import OrderedDict
from torch.utils.data.distributed import DistributedSampler

from utils import ByDomainSampler
from .client import Client
from metrics import StreamSegMetrics


def seed_worker(_):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class SiloBNClient(Client):

    def __init__(self, client_id, dataset, model, logger, writer, args, batch_size, world_size, rank, num_gpu,
                 device=None, ckpt_path=None, name=None):
        super().__init__(client_id, dataset, model, logger, writer, args, batch_size, world_size, rank, num_gpu,
                         device=device)

        self.local_test_metrics = StreamSegMetrics(args.num_classes)
        self.bn_dict = OrderedDict()
        self.name = name
        self.ckpt_path = ckpt_path

        for k, v in self.model.state_dict().items():
            if 'bn' in k:
                self.bn_dict[k] = copy.deepcopy(v)

        if args.random_seed is not None:
            g = torch.Generator()
            g.manual_seed(args.random_seed)
            self.loader = data.DataLoader(self.dataset, batch_size=self.args.dd_batch_size, worker_init_fn=seed_worker,
                                          sampler=DistributedSampler(self.dataset, num_replicas=world_size, rank=rank),
                                          num_workers=4 * num_gpu, drop_last=True, pin_memory=True, generator=g)
        else:
            self.loader = data.DataLoader(self.dataset, batch_size=self.args.dd_batch_size,
                                          sampler=DistributedSampler(self.dataset, num_replicas=world_size, rank=rank),
                                          num_workers=4 * num_gpu, drop_last=True, pin_memory=True)

        if str(self) == 'test_user_same_domain':
            self.dataset.return_domain = True
            self.bydomain_loader = \
                data.DataLoader(self.dataset, batch_size=12,
                                sampler=ByDomainSampler(self.dataset, num_replicas=world_size, rank=rank,
                                                        clients_type=self.args.clients_type,
                                                        setting_type=self.args.setting_type), num_workers=4 * num_gpu,
                                drop_last=True, pin_memory=True)
            self.dataset.return_domain = False

    def save_bn_stats(self):

        for k, v in self.model.state_dict().items():
            if 'bn' in k:
                self.bn_dict[k] = copy.deepcopy(v)

        path = os.path.join(self.ckpt_path, self.name + "_bn", self.id + '_bn.ckpt')
        torch.save(self.bn_dict, path)
        self.logger.save(os.path.join(self.ckpt_path, self.name + "_bn", self.id + '_bn.ckpt'))

    def switch_bn_stats_to_test(self, change_momentum=False):
        for name, layer in self.model.named_modules():
            if 'bn' in name:
                layer.training = True
                if change_momentum:
                    layer.momentum = 1.0

    def reset_bn_momentum(self):
        for name, layer in self.model.named_modules():
            if 'bn' in name:
                layer.momentum = 0.1

    def subs_bn_stats(self, domain, train_cl_bn_stats):

        if train_cl_bn_stats is not None:

            if domain[0] not in train_cl_bn_stats.keys():

                split = domain[0].split('_')
                cl_n = split[-1]

                if cl_n.isnumeric():

                    cl_n = int(cl_n)
                    type_cl = [k for k in train_cl_bn_stats.keys() if split[1] in k]
                    ns_type_cl = [int(k.split('_')[-1]) for k in type_cl if k.split('_')[-1].isnumeric()]
                    if len(ns_type_cl) > 0:
                        max_ns_type_cl = max(ns_type_cl)
                        mixed_domain = None
                    else:
                        max_ns_type_cl = None
                        mixed_domain = type_cl[0]

                    if max_ns_type_cl is not None:
                        if max_ns_type_cl != 1:
                            self.model.load_state_dict(train_cl_bn_stats[f"client_{split[1]}_{cl_n % max_ns_type_cl}"],
                                                       strict=False)
                        else:
                            self.model.load_state_dict(train_cl_bn_stats[f"client_{split[1]}_1"], strict=False)

                    else:
                        self.model.load_state_dict(train_cl_bn_stats[mixed_domain], strict=False)

                else:
                    type_cl = domain[0].split('_')[1]
                    dom = [k for k in train_cl_bn_stats.keys() if type_cl in k][-1]
                    self.model.load_state_dict(train_cl_bn_stats[dom], strict=False)

            else:
                self.model.load_state_dict(train_cl_bn_stats[domain[0]], strict=False)

    def copy_bn_stats(self):
        bn_dict_tmp = OrderedDict()
        for k, v in self.model.state_dict().items():
            if 'bn' in k:
                bn_dict_tmp[k] = copy.deepcopy(v)
        return bn_dict_tmp

    def test(self, metrics, ret_samples_ids=None, silobn_type='', train_cl_bn_stats=None, loader=None):
        if str(self) == 'test_user_diff_domain' or str(self) == 'test_user' or silobn_type == '_standard':
            return super().test(metrics, ret_samples_ids=ret_samples_ids, silobn_type='_standard')
        if self.args.clients_type == 'heterogeneous':
            if silobn_type == '_by_domain':
                self.dataset.return_domain = True
                bn_dict_tmp = self.copy_bn_stats()
                out = super().test(metrics, ret_samples_ids=ret_samples_ids, train_cl_bn_stats=train_cl_bn_stats,
                                   silobn_type='_by_domain', loader=self.bydomain_loader)
                self.model.load_state_dict(bn_dict_tmp, strict=False)
                self.dataset.return_domain = False
                return out
            raise NotImplementedError
        raise NotImplementedError
