import argparse

DATASETS = ['cityscapes', 'idda']
CL_TYPE = ['heterogeneous', 'uniform', 'class_imbalance']
SETTING_TYPE = ['country', 'rainy', 'bus']
POLICIES = ['poly', 'step', 'None']
MODELS = ['bisenetv2']
OPTIMIZERS = ['SGD', 'Adam']
FRAMEWORKS = ['federated', 'centralized']
ALGORITHMS = ['FedAvg', 'SiloBN']
DOM_GEN = ['cfsi', 'lab']
SERVER_OPTS = ['SGD', 'Adam', 'AdaGrad', 'FedAvgm']


def modify_command_options(args):
    if args.dataset == 'cityscapes':
        args.num_classes = 19
    elif args.dataset == 'idda' and args.remap:
        args.num_classes = 16
    elif args.dataset == 'idda' and not args.remap:
        args.num_classes = 23
    args.total_batch_size = len(args.device_ids) * args.batch_size
    args.device_ids = [int(device_id) for device_id in args.device_ids]
    args.n_devices = len(args.device_ids)
    return args


def parse_args():
    parser = argparse.ArgumentParser()

    # ||| Framework alternatives |||
    parser.add_argument('--framework', type=str, choices=FRAMEWORKS, required=True, help='Type of framework')

    # ||| Distributed and GPU options |||
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--device_ids', default=[0], nargs='+', help='GPU ids for multigpu mode')

    # ||| Reproducibility Options |||
    parser.add_argument('--random_seed', type=int, required=False, help='random seed')

    # ||| Dataset Options |||
    parser.add_argument('--dataset', type=str, choices=DATASETS, required=True, help='Name of dataset')
    parser.add_argument('--remap', action='store_true', default=False, help='Whether to remap IDDA as Cityscapes or'
                                                                            'not')
    parser.add_argument('--double_dataset', action='store_true', default=False,
                        help='Whether to double the datasets of the clients or not')
    parser.add_argument('--quadruple_dataset', action='store_true', default=False,
                        help='Whether to quadruplicate the datasets of the clients or not')
    parser.add_argument('--clients_type', type=str, choices=CL_TYPE, required=False, default='heterogeneous',
                        help='Clients distribution type')
    parser.add_argument('--setting_type', type=str, choices=SETTING_TYPE, required=False, default='',
                        help='IDDA setting types')
    parser.add_argument('--dom_gen', type=str, choices=DOM_GEN, default=None,
                        help='whether to use federated domain generalization')

    # ||| Model Options |||
    parser.add_argument('--model', type=str, default='bisenetv2', choices=MODELS,
                        help='model type')
    parser.add_argument('--hnm', action='store_true', default=False, help='Use hnm or not')
    parser.add_argument('--output_aux', action='store_true', default=False,
                        help='output_aux for bisenetv2')
    parser.add_argument('--pretrained', action='store_true', default=False,
                        help='use pretrained version of Bisenetv2')
    parser.add_argument('--bn_layer', action='store_true', default=False,
                        help='In FedBN, do not share the whole bn layer instead of only the statistics')

    # ||| Federated Algorithm Options |||
    parser.add_argument('--server_opt', help='server optimizer', choices=SERVER_OPTS, required=False)
    parser.add_argument('--algorithm', type=str, default='FedAvg', choices=ALGORITHMS,
                        help='which federated algorithm to use')
    parser.add_argument('--server_lr', type=float, default=1, help='learning rate for server optimizers')
    parser.add_argument('--server_momentum', type=float, default=0, help='momentum for server optimizers')

    # ||| Training and Testing Options |||
    parser.add_argument('--num_rounds', type=int, default=-1, help='number of rounds to simulate')
    parser.add_argument('--clients_per_round', type=int, default=-1, help='number of clients trained per round')
    parser.add_argument('--num_epochs', type=int, default=1, help='number of epochs when clients train on data')
    parser.add_argument('--batch_size', type=int, default=10, help='batch size when clients train on data')
    parser.add_argument('--test_batch_size', type=int, default=1, help='batch size when clients test on data')
    parser.add_argument('--eval_interval', type=int, default=1, help='epoch interval for eval')
    parser.add_argument('--test_interval', type=int, default=1, help='epoch interval for test')
    parser.add_argument('--mixed_precision', action='store_true', default=False,
                        help='Whether to use mixed precision or not')
    parser.add_argument('--test_diff_dom', action='store_true', default=False,
                        help='Whether to test_on_diff_dom with SiloBN')
    parser.add_argument('--dd_batch_size', type=int, default=None, help='batch size for diff dom in SiloBN')

    # ||| Transform Options |||
    parser.add_argument('--min_scale', type=float, default=0.25, help='define the lowest value for scale')
    parser.add_argument('--max_scale', type=float, default=2.0, help='define the highest value for scale')
    parser.add_argument('--h_resize', type=int, default=512, help='define the resize value for image H ')
    parser.add_argument('--w_resize', type=int, default=1024, help='define the resize value for image W ')
    parser.add_argument('--use_test_resize', action='store_true', default=False, help='whether to use test resize')
    parser.add_argument('--jitter', action='store_true', default=False, help='whether to use color jitter')
    parser.add_argument('--cv2_transform', action='store_true', default=False, help='whether to use cv2_transforms')
    parser.add_argument('--rrc_transform', action='store_true', default=False,
                        help='whether to use random resized crop')
    parser.add_argument('--rsrc_transform', action='store_true', default=False,
                        help='whether to use random scale random crop')
    parser.add_argument('--cts_norm', action='store_true', default=False,
                        help='whether to use cts normalization otherwise 0.5 for mean and std')

    # Optimizer
    parser.add_argument('--optimizer', type=str, default='SGD', choices=OPTIMIZERS, help='optimizer type')
    parser.add_argument('--lr', type=float, default=0.007, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=0, help='weight decay for the client optimizer')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum for SGD')
    parser.add_argument('--nesterov', action='store_true', default=False, help='whether to use momentum or not')
    parser.add_argument('--warmup_iters', type=int, default=1000, help='number of warmup iterations')
    parser.add_argument('--custom_lr_param', action='store_true', default=False,
                        help='Use custom lr for different params')

    # Scheduler
    parser.add_argument('--lr_policy', type=str, default='poly', choices=POLICIES, help='lr schedule policy')
    parser.add_argument('--lr_power', type=float, default=0.9, help='power for polyLR')
    parser.add_argument('--lr_decay_step', type=int, default=5000, help='decay step for stepLR')
    parser.add_argument('--lr_decay_factor', type=float, default=0.1, help='decay factor for stepLR')

    # ||| Logging Options |||
    parser.add_argument('--name', type=str, default='Experiment', help='name of the experiment')
    parser.add_argument('--print_interval', type=int, default=10, help='print interval of loss')
    parser.add_argument('--debug', action='store_true', default=False, help='verbose option')
    parser.add_argument('--save_samples', action='store_true', default=False, help='Save samples pictures on cloud')
    parser.add_argument('--wandb_offline', action='store_true', default=False,
                        help='if you want wandb offline set to True, otherwise it uploads results on cloud')
    parser.add_argument('--wandb_entity', type=str, default='feddrive', help='name of the wandb entity')

    # ||| Test and Checkpoint options |||
    parser.add_argument('--load', action='store_true', default=False, help='Whether to use pretrained or not')
    parser.add_argument('--wandb_id', type=str, required=False, help='wandb id to resume run')

    # ||| Other options |||
    parser.add_argument('--ignore_warnings', action='store_true', default=False, help='ignore all the warnings if set')
    parser.add_argument('--stop_epoch_at_step', type=int, default=-1, help='stop the epoch before the end')
    parser.add_argument('--avg_last_100', action='store_true', default=False,
                        help='compute avg and std last 100 rounds for each test type')

    return parser
