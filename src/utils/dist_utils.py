import torch
from torch import distributed


def find_free_port():
    """ https://stackoverflow.com/questions/1365265/on-localhost-how-do-i-pick-a-free-port-number """
    import socket
    from contextlib import closing

    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(('', 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return str(s.getsockname()[1])


def wait():
    distributed.barrier()


def setup():
    distributed.init_process_group(backend='nccl', init_method='env://')


def initialize_distributed(device_ids, local_rank):
    setup()
    device = torch.device(device_ids[local_rank])
    rank = distributed.get_rank()
    world_size = distributed.get_world_size()
    torch.cuda.set_device(device_ids[local_rank])

    return device, rank, world_size


def cleanup():
    distributed.destroy_process_group()
