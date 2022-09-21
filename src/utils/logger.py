import pytorch_lightning
import wandb
from pytorch_lightning.utilities import rank_zero_only

class CustomWandbLogger(pytorch_lightning.loggers.WandbLogger):
    def __init__(self, name, project, group, entity, offline=True, resume="allow", wid=None):
        super(CustomWandbLogger, self).__init__(name=name, project=project, group=group, entity=entity, offline=offline,
                                                resume=resume, id=wid)

    @rank_zero_only
    def save(self, obj):
        return wandb.save(obj)

    @staticmethod
    def restore(name, run_path, root):
        return wandb.restore(name=name, run_path=run_path, root=root)


def get_job_name(args):
    job_name = ""
    if args.framework == 'federated':
        if args.algorithm == 'SiloBN':
            job_name += "SBN_"
        job_name += f"cl{args.clients_per_round}_e{args.num_epochs}_"
    if args.dom_gen is not None:
        job_name += f"{args.dom_gen}_"
    if args.dd_batch_size:
        job_name += f"ddbs{args.dd_batch_size}_"
    if args.cv2_transform:
        job_name += "cv2_"
    if args.jitter:
        job_name += "jitter_"
    if args.use_test_resize:
        job_name += "testResize_"

    job_name += f"lr{args.lr}_rs{args.random_seed}_{args.clients_type}"
    if args.framework == 'federated':
        job_name += f"_SerOpt:{args.server_opt}_lr{args.server_lr}_m{args.server_momentum}"

    if args.custom_lr_param:
        job_name += "_customlrparam"
    if args.dataset == 'idda':
        job_name += f"_{args.setting_type}"
    job_name += f"_{args.name}"

    return job_name


def get_group_name(args):
    group_name = args.algorithm
    if args.dom_gen is not None:
        group_name += f"_{args.dom_gen}"
    return group_name


def get_project_name(framework, dataset):
    return f"{framework}_{dataset}"
