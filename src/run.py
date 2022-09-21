import time
import importlib
import warnings
from utils import parse_args, modify_command_options


def run_experiment():
    if args.framework == 'federated':
        main_module = 'fed_setting.main'
        main = getattr(importlib.import_module(main_module), 'main')
        main(args)
    elif args.framework == 'centralized':
        main_module = 'centr_setting.main'
        main = getattr(importlib.import_module(main_module), 'main')
        main(args)
    else:
        raise NotImplementedError


if __name__ == '__main__':

    start = time.time()

    parser = parse_args()
    args = parser.parse_args()
    args = modify_command_options(args)

    if args.ignore_warnings:
        warnings.filterwarnings("ignore")

    run_experiment()

    end = time.time()
    print(f"Elapsed time: {round(end - start, 2)}")
