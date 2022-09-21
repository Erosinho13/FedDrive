# --------------------------------------
# Centralized Training
# --------------------------------------

import os
import torch.nn as nn
import torch.optim as optim

from utils import dist_utils, setup_env, load_from_checkpoint, setup_pre_training, set_params
from utils import get_scheduler
from utils import setup_clients
from modules import make_model
from metrics import print_stats


def main(args):
    writer, device, rank, world_size, logger, label2color, denorm = setup_env(args)

    # ===== Create model ===== #

    writer.write("Create model and data")
    model_train = make_model(args)
    model_train = model_train.to(device)
    trainable_params = set_params(model_train, args)
    model_train = nn.parallel.DistributedDataParallel(model_train, device_ids=[args.device_ids[args.local_rank]],
                                                      output_device=args.device_ids[args.local_rank],
                                                      find_unused_parameters=False)
    # ===== Setup metrics and training ===== #

    train_metrics, val_metrics, miou, acc, ckpt_path = setup_pre_training(writer, args.num_classes, args.framework,
                                                                          args.dataset, args.name)
    # ----- Create datasets and dataloader ----- #

    train_clients, test_clients = setup_clients(args, logger, writer, model_train, world_size, rank, args.n_devices,
                                                device, ckpt_path)
    writer.write("Done.")

    # ====== Setup optimizer and scheduler ====== #
    if args.optimizer == 'SGD':
        optimizer = optim.SGD(trainable_params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay,
                              nesterov=args.nesterov)
    else:
        optimizer = optim.Adam(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = get_scheduler(args, optimizer, max_iter=args.num_epochs * train_clients[0].len_loader())

    # ====== Load from checkpoint ====== #
    checkpoint_epoch = 0
    if args.load:  # load model from checkpoint
        checkpoint_epoch = load_from_checkpoint(writer, logger, args.framework, args.dataset, args.name, ckpt_path,
                                                args.wandb_id, model_train, args.wandb_entity, optimizer=optimizer,
                                                scheduler=scheduler)

    for e in range(checkpoint_epoch, args.num_epochs):
        # =====  Train  =====
        writer.write(f"EPOCH: {e + 1}/{args.num_epochs}")

        model_train.train()
        _ = train_clients[0].run_epoch(e, train_metrics, optimizer, scheduler)

        train_metrics.synch(device)

        if args.local_rank == 0:
            train_score = train_metrics.get_results()
            logger.log_metrics({'Train Mean IoU': train_score['Mean IoU']}, step=e + 1)

        train_metrics.reset()

        # =====  Validation  =====
        if (e + 1) % args.test_interval == 0 or (e + 1) == args.num_epochs:
            miou = print_stats(args, e + 1, test_clients, val_metrics, logger, writer, ret_score='miou',
                               action='test', label2color=label2color, denorm=denorm)
            writer.write(f"Test mIoU on epoch {e + 1}: {miou}")

        # =====  Save Model  =====
        train_clients[0].save_model(e + 1, os.path.join(ckpt_path, '{}.ckpt'.format(args.name)), optimizer, scheduler)
        logger.save(os.path.join(ckpt_path, '{}.ckpt'.format(args.name)))

    if args.dataset == 'idda':
        writer.write(f"Final mIoU: \n diff_dom = {miou[0]} \n same_dom = {miou[1]}")
    else:
        writer.write(f"Final mIou: {miou}")

    dist_utils.cleanup()
