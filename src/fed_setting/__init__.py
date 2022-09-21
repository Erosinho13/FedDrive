from .server import Server
from .silobnserver import SiloBNServer


def define_server(client_model, logger, writer, args, train_clients, device):
    if args.algorithm == 'FedAvg':
        server_ = Server(client_model, logger, writer, args.local_rank, args.server_lr, args.server_momentum,
                         args.server_opt)
    elif args.algorithm == 'SiloBN':
        server_ = SiloBNServer(client_model, logger, writer, args.local_rank, args.server_lr, args.server_momentum,
                               args.server_opt, bn_layer=args.bn_layer, train_clients=train_clients)
    else:
        raise NotImplementedError

    return server_
