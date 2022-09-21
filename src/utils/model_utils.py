import os
import copy
import torch
from utils import get_project_name


def zero_copy(model):
    tmp_model = copy.deepcopy(model)
    for tp in tmp_model.parameters():
        tp.data = torch.zeros_like(tp.data)
    return tmp_model


def load_from_checkpoint(writer, logger, framework, dataset, name, ckpt_path, wandb_id, model, entity, optimizer=None,
                         scheduler=None):
    writer.write("--- Loading model from checkpoint ---")

    load_path = os.path.join(ckpt_path, name + '.ckpt')
    run_path = os.path.join(entity.lower(), get_project_name(framework, dataset), wandb_id)
    logger.restore(name=load_path, run_path=run_path, root=".")
    checkpoint = torch.load(load_path)
    model.load_state_dict(checkpoint["model_state"])
    if framework == 'federated':
        checkpoint_step = checkpoint["round"]
        writer.write(f"[!] Model restored from round {checkpoint['round']}")
        if 'last_scores' in checkpoint.keys():
            last_scores = checkpoint['last_scores']
            writer.write("Done.")
            return checkpoint_step, last_scores
    elif framework == 'centralized':
        checkpoint_step = checkpoint["epoch"]
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        writer.write(f"[!] Model restored from epoch {checkpoint['epoch']}")
    else:
        raise NotImplementedError

    writer.write("Done.")

    return checkpoint_step, None


def load_server_optim(writer, logger, framework, dataset, name, ckpt_path, wandb_id, server, entity):
    writer.write("--- Loading server_optimizer from checkpoint ---")

    load_path = os.path.join(ckpt_path, name + '_server_opt.ckpt')
    run_path = os.path.join(entity.lower(), get_project_name(framework, dataset), wandb_id)
    logger.restore(name=load_path, run_path=run_path, root=".")
    server_opt_state = torch.load(load_path)
    server.optimizer.load_state_dict(server_opt_state)


def load_client_bn(writer, logger, framework, dataset, name, ckpt_path, wandb_id, train_clients, entity):
    writer.write("--- Loading clients bn statistics from checkpoint ---")

    for c in train_clients:
        load_path = os.path.join(ckpt_path, name + "_bn", c.id + '_bn.ckpt')
        run_path = os.path.join(entity.lower(), get_project_name(framework, dataset), wandb_id)
        logger.restore(name=load_path, run_path=run_path, root=".")

        c.bn_dict = torch.load(load_path)


def set_params(model, args):
    if args.custom_lr_param:
        if hasattr(model, 'get_params'):
            print("correct")
            wd_params, nowd_params, lr_mul_wd_params, lr_mul_nowd_params = model.get_params()
            wd_val = 0
            params_list = [
                {'params': wd_params, },
                {'params': nowd_params, 'weight_decay': wd_val},
                {'params': lr_mul_wd_params, 'lr': args.lr * 10},
                {'params': lr_mul_nowd_params, 'weight_decay': wd_val, 'lr': args.lr * 10},
            ]
        else:
            wd_params, non_wd_params = [], []
            for name, param in model.named_parameters():
                if param.dim() == 1:
                    non_wd_params.append(param)
                elif param.dim() == 2 or param.dim() == 4:
                    wd_params.append(param)
            params_list = [
                {'params': wd_params, },
                {'params': non_wd_params, 'weight_decay': 0},
            ]
    else:
        params_list = [{"params": filter(lambda p: p.requires_grad, model.parameters()),
                        'weight_decay': args.weight_decay}]
    return params_list
