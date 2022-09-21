import copy
import torch

from fed_setting import Server
from collections import OrderedDict


class SiloBNServer(Server):

    def __init__(self, model, logger, writer, local_rank, lr, momentum, optimizer=None, bn_layer=False,
                 train_clients=None):
        super().__init__(model, logger, writer, local_rank, lr, momentum, optimizer)
        self.bn_layer = bn_layer
        self.train_clients = train_clients

    def load_server_model_on_client(self, client):
        # update clients model except for bn
        client.model.load_state_dict(client.bn_dict, strict=False)
        for k, v in self.model_params_dict.items():
            if self.bn_layer:
                if 'bn' not in k:
                    client.model.state_dict()[k].data.copy_(v)
            else:
                if 'bn.running' not in k and 'bn.num_batches_tracked' not in k:
                    client.model.state_dict()[k].data.copy_(v)

    def _aggregation(self):

        total_weight = 0.
        base = OrderedDict()

        for (client_samples, client_model) in self.updates:

            total_weight += client_samples
            for key, value in client_model.items():
                if self.bn_layer:
                    if 'bn' not in key:
                        if key in base:
                            base[key] += client_samples * value.type(torch.FloatTensor)
                        else:
                            base[key] = client_samples * value.type(torch.FloatTensor)
                else:
                    if 'bn.running' not in key and 'bn.num_batches_tracked' not in key:
                        if key in base:
                            base[key] += client_samples * value.type(torch.FloatTensor)
                        else:
                            base[key] = client_samples * value.type(torch.FloatTensor)
        averaged_sol_n = copy.deepcopy(self.model_params_dict)
        for key, value in base.items():
            if total_weight != 0:
                averaged_sol_n[key] = value.to('cuda') / total_weight

        return averaged_sol_n

    def get_cl_bn_stats(self):
        train_cl_bn_stats = {}
        for c in self.train_clients:
            train_cl_bn_stats[str(c)] = OrderedDict([(k, v) for k, v in c.bn_dict.items() if 'running' in k])
        return train_cl_bn_stats
