import numpy as np
import torch
import copy
import torch.optim as optim

from collections import OrderedDict


class Server:

    def __init__(self, model, logger, writer, local_rank, lr, momentum, optimizer=None):
        self.model = copy.deepcopy(model)
        self.model_params_dict = copy.deepcopy(self.model.state_dict())
        self.logger = logger
        self.writer = writer
        self.selected_clients = []
        self.updates = []
        self.local_rank = local_rank
        self.opt_string = optimizer
        self.lr = lr
        self.momentum = momentum
        self.optimizer = self._get_optimizer()
        self.total_grad = 0

    def _get_optimizer(self):

        if self.opt_string is None:
            self.writer.write("Running without server optimizer")
            return None

        if self.opt_string == 'SGD':
            return optim.SGD(params=self.model.parameters(), lr=self.lr, momentum=self.momentum)

        if self.opt_string == 'FedAvgm':
            return optim.SGD(params=self.model.parameters(), lr=1, momentum=0.9)

        if self.opt_string == 'Adam':
            return optim.Adam(params=self.model.parameters(), lr=self.lr, betas=(0.9, 0.99), eps=10 ** (-1))

        if self.opt_string == 'AdaGrad':
            return optim.Adagrad(params=self.model.parameters(), lr=self.lr, eps=10 ** (-2))

        raise NotImplementedError

    def select_clients(self, my_round, possible_clients, num_clients=20):
        num_clients = min(num_clients, len(possible_clients))
        np.random.seed(my_round)
        self.selected_clients = np.random.choice(possible_clients, num_clients, replace=False)

    def call_training(self, c, metrics):
        out = c.train(metrics=metrics)
        return out

    def add_updates(self, num_samples, update):
        self.updates.append((num_samples, update))

    def _compute_client_delta(self, cmodel):
        delta = OrderedDict.fromkeys(cmodel.keys())
        for k, x, y in zip(self.model_params_dict.keys(), self.model_params_dict.values(), cmodel.values()):
            delta[k] = y - x if "running" not in k and "num_batches_tracked" not in k else y
        return delta

    def load_server_model_on_client(self, client):
        client.model.load_state_dict(self.model_params_dict)

    def train_model(self, metrics):

        if self.optimizer is not None:
            self.optimizer.zero_grad()

        clients = self.selected_clients
        losses = {}

        for i, c in enumerate(clients):

            self.writer.write(f"CLIENT {i + 1}/{len(clients)}: {c}")

            self.load_server_model_on_client(c)
            out = self.call_training(c, metrics)
            c.save_bn_stats()

            if self.local_rank == 0:
                num_samples, update, dict_losses_list = out
                losses[c.id] = {'loss': dict_losses_list, 'num_samples': num_samples}
            else:
                num_samples, update = out

            if self.optimizer is not None:
                update = self._compute_client_delta(update)

            self.add_updates(num_samples=num_samples, update=update)

        if self.local_rank == 0:
            return losses
        return None

    def _server_opt(self, pseudo_gradient):

        for n, p in self.model.named_parameters():
            p.grad = -1.0 * pseudo_gradient[n]

        self.optimizer.step()

        bn_layers = OrderedDict(
            {k: v for k, v in pseudo_gradient.items() if "running" in k or "num_batches_tracked" in k})
        self.model.load_state_dict(bn_layers, strict=False)

    def _aggregation(self):
        total_weight = 0.
        base = OrderedDict()

        for (client_samples, client_model) in self.updates:

            total_weight += client_samples
            for key, value in client_model.items():
                if key in base:
                    base[key] += client_samples * value.type(torch.FloatTensor)
                else:
                    base[key] = client_samples * value.type(torch.FloatTensor)
        averaged_sol_n = copy.deepcopy(self.model_params_dict)
        for key, value in base.items():
            if total_weight != 0:
                averaged_sol_n[key] = value.to('cuda') / total_weight

        return averaged_sol_n

    def _get_model_total_grad(self):
        total_norm = 0
        for name, p in self.model.named_parameters():
            if p.requires_grad:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_grad = total_norm ** 0.5
        self.writer.write(f"total grad norm: {round(total_grad, 2)}")  # 0: no gradients server side
        return total_grad

    def update_model(self):
        """FedAvg on the clients' updates for the current round.
        Weighted average of self.updates, where the weight is given by the number
        of samples seen by the corresponding client at training time.
        Saves the new central model in self.client_model and its state dictionary in self.model
        """

        averaged_sol_n = self._aggregation()

        if self.optimizer is not None:  # optimizer step
            self._server_opt(averaged_sol_n)
            self.total_grad = self._get_model_total_grad()
        else:
            self.model.load_state_dict(averaged_sol_n, strict=False)
        self.model_params_dict = copy.deepcopy(self.model.state_dict())

        self.updates = []

    def get_cl_bn_stats(self):
        pass

    def test_model(self, clients_to_test, metrics, ret_samples_bool=False, silobn_type=''):

        loss_test = {}
        metrics.reset()
        ret_samples_list = []

        for client in clients_to_test:

            train_cl_bn_stats = self.get_cl_bn_stats() if str(client) == 'test_user_same_domain' else None

            if silobn_type == '':
                self.writer.write(f"Testing client {client}...")
            else:
                self.writer.write(f"Testing client {client}, mode {silobn_type[1:]}...")
            ret_samples_ids = None
            if ret_samples_bool:
                # draw ret_samples_ids
                ret_samples_ids = np.random.choice(len(client.loader), 3, replace=False)

            self.load_server_model_on_client(client)
            class_loss, ret_samples = client.test(metrics, ret_samples_ids, silobn_type=silobn_type,
                                                  train_cl_bn_stats=train_cl_bn_stats)
            ret_samples_list.extend(ret_samples)
            loss_test[client.id] = {'loss': class_loss, 'num_samples': client.num_samples}

        return loss_test, ret_samples_list

    def get_clients_info(self, clients):
        if clients is None:
            clients = self.selected_clients
        num_samples = {c.id: c.num_samples for c in clients}
        return num_samples

    def save_model(self, rounds, path, opt_path, last_scores=None):
        state = {
            "round": rounds,
            "model_state": self.model_params_dict
        }
        if last_scores is not None:
            lsc = dict(last_scores)
            for k, v in lsc.items():
                lsc[k] = dict(v)
            state['last_scores'] = lsc
        torch.save(state, path)
        if self.optimizer is not None:
            torch.save(self.optimizer.state_dict(), opt_path)
        return path

    @staticmethod
    def num_parameters(params):
        return sum(p.numel() for p in params if p.requires_grad)

    @staticmethod
    def online(clients):
        """We assume all users are always online."""
        return clients
