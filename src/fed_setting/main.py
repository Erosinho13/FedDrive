# --------------------------------------
# Federated Training
# --------------------------------------

import os
from collections import defaultdict

import numpy as np
import torch.nn as nn

from utils import dist_utils, setup_env, load_from_checkpoint, setup_pre_training, load_server_optim, load_client_bn
from utils import setup_clients
from modules import make_model
from utils import weight_train_loss
from metrics import print_stats
from fed_setting import define_server


def main(args):

    writer, device, rank, world_size, logger, label2color, denorm = setup_env(args)

    # ===== Create client model and share params with server model ===== #

    writer.write("Creating client model and sharing params with server model...")
    client_model = make_model(args)
    client_model = client_model.to(device)
    client_model = nn.parallel.DistributedDataParallel(client_model, device_ids=[args.device_ids[args.local_rank]],
                                                       output_device=args.device_ids[args.local_rank],
                                                       find_unused_parameters=False)

    train_metrics, val_metrics, miou, acc, ckpt_path = setup_pre_training(writer, args.num_classes, args.framework,
                                                                          args.dataset, args.name, args.algorithm)

    last_scores = defaultdict(lambda: defaultdict(lambda: []))

    checkpoint_round = 0
    if args.load:  # load model from checkpoint
        checkpoint_round, last_scores_tmp = load_from_checkpoint(writer, logger, args.framework, args.dataset,
                                                                 args.name, ckpt_path, args.wandb_id, client_model,
                                                                 args.wandb_entity)
        if last_scores_tmp is not None:
            last_scores = last_scores_tmp

    train_clients, test_clients = setup_clients(args, logger, writer, client_model, world_size, rank, args.n_devices,
                                                device, ckpt_path)

    server = define_server(client_model, logger, writer, args, train_clients, device)
    writer.write("Done")

    if args.load and args.server_opt is not None:
        load_server_optim(writer, logger, args.framework, args.dataset, args.name, ckpt_path, args.wandb_id, server,
                          args.wandb_entity)
    if args.load and args.algorithm == 'SiloBN':
        load_client_bn(writer, logger, args.framework, args.dataset, args.name, ckpt_path, args.wandb_id, train_clients,
                       args.wandb_entity)

    for r in range(checkpoint_round, args.num_rounds):
        # =====  Train  =====

        # start_round = time.time()
        writer.write(
            f"ROUND {r + 1}/{args.num_rounds}: Training {args.clients_per_round} Clients...")
        server.select_clients(r, server.online(train_clients), num_clients=args.clients_per_round)
        losses = server.train_model(metrics=train_metrics)

        if args.local_rank == 0:
            train_score = train_metrics.get_results()
            writer.write(f"Mean IoU train data: {'%.3f' % train_score['Mean IoU']}\n")
            logger.log_metrics({'Partial Train Mean IoU': train_score['Mean IoU']}, step=r + 1)
            logger.log_metrics({'Partial Train Overall Accuracy': train_score['Overall Acc']}, step=r + 1)

            # Get losses weighted over round clients
            round_losses = weight_train_loss(losses)

            for name, l in round_losses.items():
                logger.log_metrics({f"R-{name}": l}, step=r + 1)

        train_metrics.reset()
        server.update_model()

        # Save server model
        opt_path = os.path.join(ckpt_path, '{}_server_opt.ckpt'.format(args.name))
        server.save_model(r + 1, os.path.join(ckpt_path, '{}.ckpt'.format(args.name)), opt_path,
                          last_scores=last_scores if args.avg_last_100 else None)
        logger.save(os.path.join(ckpt_path, f"{args.name}.ckpt"))
        logger.save(os.path.join(ckpt_path, f"{args.name}_server_opt.ckpt"))

        if (r + 1) >= args.num_rounds - 100 and args.avg_last_100 and (r + 1) % 5 == 0:
            scores = print_stats(args, r + 1, test_clients, val_metrics, logger, writer,
                                 ret_score='miou', action='test', server=server,
                                 train_clients=train_clients, label2color=label2color, denorm=denorm,
                                 last=True)
            for client, v in scores.items():
                for k in v.keys():
                    strategy = 'standard' if k == '' else k[1:]
                    writer.write(f"Test mIoU on round {r + 1} for client {client} with strategy {strategy}: "
                                 f"{round(scores[client][k]['mIoU'] * 100, 2)}%")
                    if type(last_scores) == dict:
                        if client not in last_scores.keys():
                            last_scores[client] = defaultdict(lambda: [])
                    last_scores[client][strategy].append(scores[client][k]['mIoU'])

        elif (r + 1) % args.test_interval == 0 or (r + 1) == args.num_rounds:
            scores = print_stats(args, r + 1, test_clients, val_metrics, logger, writer,
                                 ret_score='miou', action='test', server=server,
                                 train_clients=train_clients, label2color=label2color, denorm=denorm)
            for client, v in scores.items():
                for k in v.keys():
                    strategy = 'standard' if k == '' else k[1:]
                    writer.write(f"Test mIoU on round {r + 1} for client {client} with strategy {strategy}: "
                                 f"{round(scores[client][k]['mIoU'] * 100, 2)}%")

    if len(last_scores.items()) > 0:
        last_scores = dict(last_scores)
        for k, v in last_scores.items():
            last_scores[k] = dict(v)
            for k2, v2 in v.items():
                logger.log_metrics({
                    f'{k} {k2} last 100 rounds mean mIoU': np.mean(np.array(v2)),
                    f'{k} {k2} last 100 rounds std mIoU': np.std(np.array(v2))
                }, step=0)

    writer.write("Job completed!!")

    dist_utils.cleanup()
