import json
import os
import random
from collections import defaultdict

import numpy as np
import torch
import wandb

from torch.utils.data import DistributedSampler
from .writer import Writer
from .dist_utils import initialize_distributed
from .logger import CustomWandbLogger, get_job_name, get_project_name, get_group_name
from .data_utils import Label2Color, color_map, Denormalize
from metrics import StreamSegMetrics


def setup_env(args):
    # ===== Setup Writer ===== #
    writer = Writer(args)

    # ===== Setup distributed ===== #
    writer.write("Setting up distributed...")
    device, rank, world_size = initialize_distributed(args.device_ids, args.local_rank)
    writer.write(f"Let's use {args.n_devices} GPUs!")
    writer.write(f"Done")

    # ===== Initialize wandb ===== #
    writer.write("Initializing wandb...")
    if args.load:
        ids = args.wandb_id
    else:
        ids = wandb.util.generate_id()
        args.wandb_id = ids

    logger = CustomWandbLogger(name=get_job_name(args), project=get_project_name(args.framework, args.dataset),
                               group=get_group_name(args), entity=args.wandb_entity, offline=args.wandb_offline,
                               resume="allow", wid=ids)
    logger.log_hyperparams(args)

    writer.write("Done.")

    # ===== Setup random seed to reproducibility ===== #
    if args.random_seed is not None:
        writer.write("Setting up the random seed for reproducibility")
        set_seed(args.random_seed)
        writer.write("Done")

    # ====== UTILS for ret_samples ===== #
    label2color = Label2Color(cmap=color_map(args.dataset, args.remap))  # convert labels to images
    if args.dataset == 'idda':
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]
    elif args.dataset == 'cityscapes':
        if args.cts_norm:
            mean = [0.3257, 0.3690, 0.3223]
            std = [0.2112, 0.2148, 0.2115]
        else:
            mean = [0.5, 0.5, 0.5]
            std = [0.5, 0.5, 0.5]
    else:
        mean, std = None, None
    denorm = Denormalize(mean=mean, std=std)  # de-normalization for original images

    return writer, device, rank, world_size, logger, label2color, denorm


def set_seed(random_seed):
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # torch.use_deterministic_algorithms(True)


def set_metrics(writer, num_classes):
    writer.write("Setting up metrics...")
    val_metrics = StreamSegMetrics(num_classes)
    train_metrics = StreamSegMetrics(num_classes)
    writer.write("Done.")
    return train_metrics, val_metrics


def setup_pre_training(writer, num_classes, framework, dataset, name=None, algorithm=None):

    train_metrics, val_metrics = set_metrics(writer, num_classes)
    writer.write("Simulating clients trainings...")
    if framework == 'federated':
        ckpt_path = os.path.join('checkpoints', framework, dataset, algorithm)
    else:
        ckpt_path = os.path.join('checkpoints', framework, dataset)
    if not os.path.exists(ckpt_path):
        os.makedirs(ckpt_path)
    if not os.path.exists(os.path.join(ckpt_path, name+"_bn")) and framework == 'federated':
        os.makedirs((os.path.join(ckpt_path, name+"_bn")))
    return train_metrics, val_metrics, 0, 0, ckpt_path


class ByDomainSampler(DistributedSampler):

    def __init__(self, dataset, num_replicas, rank, clients_type, setting_type):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank, shuffle=False, seed=0)
        self.dataset = dataset
        if os.path.exists(f'idda_{clients_type}_{setting_type}_idx.json'):
            with open(f'idda_{clients_type}_{setting_type}_idx.json', 'r') as f:
                self.idx_by_dom = json.load(f)
        else:
            self.idx_by_dom = defaultdict(lambda: [])
            for i, (_, _, dom) in enumerate(self.dataset):
                self.idx_by_dom[dom].append(i)
            with open(f'idda_{clients_type}_{setting_type}_idx.json', 'w') as f:
                json.dump(self.idx_by_dom, f)

    def __iter__(self):
        indices = []
        for v in self.idx_by_dom.values():
            indices += v
        return iter(indices)
