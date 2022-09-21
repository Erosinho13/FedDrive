import copy
import warnings
import torch
import torch.nn as nn
import torch.optim as optim
from torch import distributed
from collections import defaultdict
import numpy as np
import random

from utils import get_scheduler, set_params
from utils import HardNegativeMining, MeanReduction
from torch.utils.data.distributed import DistributedSampler
from torch.utils import data
from torch.cuda.amp import autocast, GradScaler


def seed_worker(_):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


class Client:

    def __init__(self, client_id, dataset, model, logger, writer, args, batch_size, world_size, rank, num_gpu,
                 device=None, **kwargs):

        self.id = client_id
        self.dataset = dataset
        self._model = model
        self.device = device
        self.batch_size = batch_size
        self.logger = logger
        self.writer = writer
        self.args = args

        if args.random_seed is not None:
            g = torch.Generator()
            g.manual_seed(args.random_seed)
            self.loader = data.DataLoader(self.dataset, batch_size=self.batch_size, worker_init_fn=seed_worker,
                                          sampler=DistributedSampler(self.dataset, num_replicas=world_size, rank=rank),
                                          num_workers=4 * num_gpu, drop_last=True, pin_memory=True, generator=g)
        else:
            self.loader = data.DataLoader(self.dataset, batch_size=self.batch_size,
                                          sampler=DistributedSampler(self.dataset, num_replicas=world_size, rank=rank),
                                          num_workers=4 * num_gpu, drop_last=True, pin_memory=True)

        self.criterion = nn.CrossEntropyLoss(ignore_index=255, reduction='none')
        self.reduction = HardNegativeMining() if args.hnm else MeanReduction()

        if self.args.mixed_precision:
            self.scaler = GradScaler()

    def save_bn_stats(self):
        pass

    def calc_losses(self, images, labels):

        if self.args.model == 'bisenetv2':
            if self.args.output_aux:
                outputs, feat2, feat3, feat4, feat5_4 = self.model(images)
                loss = self.reduction(self.criterion(outputs, labels), labels)
                boost_loss = 0
                boost_loss += self.reduction(self.criterion(feat2, labels), labels)
                boost_loss += self.reduction(self.criterion(feat3, labels), labels)
                boost_loss += self.reduction(self.criterion(feat4, labels), labels)
                boost_loss += self.reduction(self.criterion(feat5_4, labels), labels)

                loss_tot = loss + boost_loss
                dict_calc_losses = {'loss': loss, 'boost_loss': boost_loss, 'loss_tot': loss_tot}

            else:
                outputs = self.model(images)
                loss_tot = self.reduction(self.criterion(outputs, labels), labels)
                dict_calc_losses = {'loss_tot': loss_tot}

        else:
            raise NotImplementedError

        return dict_calc_losses, outputs

    def handle_grad(self, loss_tot):
        pass

    @staticmethod
    def calc_loss_fed(dict_losses):
        return dict_losses

    @staticmethod
    def update_metrics(metrics, outputs, labels):
        _, prediction = outputs.max(dim=1)
        labels = labels.cpu().numpy()
        prediction = prediction.cpu().numpy()
        metrics.update(labels, prediction)

    @staticmethod
    def print_step_loss(losses, scheduler, logger, step):
        for name, l in losses.items():
            logger.log_metrics({f"Train_{name}": l}, step=step)
        if scheduler is not None:
            logger.log_metrics({"Learning Rate": scheduler.get_last_lr()[0]}, step=step)

    @staticmethod
    def apply_loss_penalties(loss_tot):
        return loss_tot

    def clip_grad(self):
        pass

    def run_epoch(self, cur_epoch, metrics, optimizer, scheduler=None):

        dict_all_epoch_losses = defaultdict(lambda: 0)

        self.loader.sampler.set_epoch(cur_epoch)

        for cur_step, (images, labels) in enumerate(self.loader):

            if self.args.stop_epoch_at_step != -1 and cur_step > self.args.stop_epoch_at_step:
                break

            images = images.to(self.device, dtype=torch.float32)
            labels = labels.to(self.device, dtype=torch.long)

            optimizer.zero_grad()

            if self.args.mixed_precision:
                with autocast():
                    dict_calc_losses, outputs = self.calc_losses(images, labels)
                dict_calc_losses['loss_tot'] = self.apply_loss_penalties(dict_calc_losses['loss_tot'])
                self.scaler.scale(dict_calc_losses['loss_tot']).backward()
            else:
                dict_calc_losses, outputs = self.calc_losses(images, labels)
                dict_calc_losses['loss_tot'] = self.apply_loss_penalties(dict_calc_losses['loss_tot'])
                dict_calc_losses['loss_tot'].backward()

            self.handle_grad(dict_calc_losses['loss_tot'])

            if (cur_step + 1) % self.args.print_interval == 0 and self.args.framework == 'centralized':
                self.print_step_loss(dict_calc_losses, scheduler, self.logger,
                                     len(self.loader) * cur_epoch + cur_step + 1)

            dict_calc_losses = self.calc_loss_fed(dict_calc_losses)

            self.clip_grad()

            self.scaler.step(optimizer) if self.args.mixed_precision else optimizer.step()

            if scheduler is not None:
                scheduler.step()

            if self.args.framework == 'federated' and cur_epoch == self.args.num_epochs - 1:
                self.update_metrics(metrics, outputs, labels)
            elif self.args.framework == 'centralized':
                self.update_metrics(metrics, outputs, labels)

            if self.args.mixed_precision:
                self.scaler.update()

            for name, l in dict_calc_losses.items():
                if type(l) != int:
                    dict_all_epoch_losses[name] += l.detach().item()
                else:
                    dict_all_epoch_losses[name] += l

        self.writer.write(f"EPOCH {cur_epoch + 1}: ended.")
        print_string = ""
        for name, l in dict_all_epoch_losses.items():
            dict_all_epoch_losses[name] /= len(self.loader)
            print_string += f"{name}={'%.3f' % dict_all_epoch_losses[name]}, "
        self.writer.write(print_string)

        return dict_all_epoch_losses

    def generate_update(self):
        return copy.deepcopy(self.model.state_dict())

    def _configure_optimizer(self, params):
        if self.args.optimizer == 'SGD':
            optimizer = optim.SGD(params, lr=self.args.lr, momentum=self.args.momentum,
                                  weight_decay=self.args.weight_decay, nesterov=self.args.nesterov)
        else:
            optimizer = optim.Adam(params, lr=self.args.lr, weight_decay=self.args.weight_decay)
        scheduler = get_scheduler(self.args, optimizer,
                                  max_iter=10000 * self.args.num_epochs * len(self.loader))
        return optimizer, scheduler

    def handle_log_loss(self, dict_all_epoch_losses, dict_losses_list):
        for n, l in dict_all_epoch_losses.items():
            dict_all_epoch_losses[n] = torch.tensor(l).to(self.device)
            distributed.reduce(dict_all_epoch_losses[n], dst=0)
            if self.args.local_rank == 0:
                dict_losses_list[n].append(dict_all_epoch_losses[n] / distributed.get_world_size())
        return dict_all_epoch_losses, dict_losses_list

    def train(self, metrics):

        params = set_params(self.model, self.args)
        num_train_samples = len(self.dataset)

        optimizer, scheduler = self._configure_optimizer(params)

        dict_losses_list = defaultdict(lambda: [])
        self.model.train()

        for epoch in range(self.args.num_epochs):
            dict_all_epoch_losses = self.run_epoch(epoch, metrics, optimizer, scheduler)
            dict_all_epoch_losses, dict_losses_list = self.handle_log_loss(dict_all_epoch_losses, dict_losses_list)

        metrics.synch(self.device)

        if self.args.framework == 'federated':
            update = self.generate_update()
        else:
            update = None

        if self.args.local_rank == 0:
            return num_train_samples, update, dict_losses_list
        return num_train_samples, update

    def switch_bn_stats_to_test(self, change_momentum=False):
        pass

    def reset_bn_momentum(self):
        pass

    def subs_bn_stats(self, domain, train_cl_bn_stats):
        pass

    def copy_bn_stats(self):
        pass

    def test(self, metrics, ret_samples_ids=None, silobn_type=None, train_cl_bn_stats=None, loader=None):

        self.model.eval()

        # idda diff_dom + idda same_dom standard
        if silobn_type == '' or silobn_type == '_standard':
            self.switch_bn_stats_to_test()

        bn_dict_tmp = None

        class_loss = 0.0
        ret_samples = []

        if loader is None:
            loader = self.loader

        with torch.no_grad():
            for i, sample in enumerate(loader):

                if self.args.stop_epoch_at_step != -1 and i > self.args.stop_epoch_at_step:
                    break

                self.writer.write(f'{self}: {i + 1}/{len(loader)}, {round((i + 1) / len(loader) * 100, 2)}%')

                if self.args.dataset == 'idda':
                    # idda heterogeneous same_dom by_domain
                    if self.dataset.return_domain:
                        images, labels, domain = sample
                    else:
                        images, labels = sample
                        domain = None
                else:
                    images, labels = sample

                # idda heterogeneous same_dom by_domain
                if self.args.dataset == 'idda':
                    self.subs_bn_stats(domain, train_cl_bn_stats)

                if self.args.model == 'bisenetv2':
                    original_images, images = images
                else:
                    original_images = images

                images = images.to(self.device, dtype=torch.float32)
                labels = labels.to(self.device, dtype=torch.long)

                outputs = self.model(images, test=True, use_test_resize=self.args.use_test_resize) \
                    if self.args.model == 'bisenetv2' else self.model(images)

                loss = self.reduction(self.criterion(outputs, labels), labels)
                class_loss += loss.item()

                _, prediction = outputs.max(dim=1)
                labels = labels.cpu().numpy()
                prediction = prediction.cpu().numpy()
                metrics.update(labels, prediction)

                if ret_samples_ids is not None and i in ret_samples_ids:  # get samples
                    ret_samples.append((original_images[0].detach().cpu().numpy(),
                                        labels[0], prediction[0]))

                if bn_dict_tmp is not None:
                    self.model.load_state_dict(bn_dict_tmp, strict=False)

            metrics.synch(self.device)

            class_loss = torch.tensor(class_loss).to(self.device)
            distributed.reduce(class_loss, dst=0)

            class_loss = class_loss / distributed.get_world_size() / len(self.loader)

        return class_loss, ret_samples

    def save_model(self, epochs, path, optimizer, scheduler):
        state = {
            "epoch": epochs,
            "model_state": self.model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict()}
        torch.save(state, path)
        return path

    def __str__(self):
        return self.id

    @property
    def num_samples(self):
        return len(self.dataset)

    def len_loader(self):
        return len(self.loader)

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, model):
        warnings.warn('The current implementation shares the model among all clients.'
                      'Setting it on one client will effectively modify all clients.')
        self._model = model
