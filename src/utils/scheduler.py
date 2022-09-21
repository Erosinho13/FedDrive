import torch


def get_scheduler(opts, optimizer, max_iter=None):
    if opts.lr_policy == 'poly':
        assert max_iter is not None, "max_iter necessary for poly LR scheduler"
        return torch.optim.lr_scheduler.LambdaLR(optimizer,
                                                 lr_lambda=lambda cur_iter: (1 - cur_iter / max_iter) ** opts.lr_power)
    if opts.lr_policy == 'step':
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=opts.lr_decay_step, gamma=opts.lr_decay_factor)

    return None


class WarmupLrScheduler(torch.optim.lr_scheduler._LRScheduler):

    def __init__(self, optimizer, warmup_iter=500, warmup_ratio=5e-4, warmup='exp', last_epoch=-1):
        super().__init__(optimizer, last_epoch)
        self.warmup_iter = warmup_iter
        self.warmup_ratio = warmup_ratio
        self.warmup = warmup

    def get_lr(self):
        ratio = self.get_lr_ratio()
        lrs = [ratio * lr for lr in self.base_lrs]
        return lrs

    def get_lr_ratio(self):
        ratio = self.get_warmup_ratio() if self.last_epoch < self.warmup_iter else self.get_main_ratio()
        return ratio

    def get_main_ratio(self):
        raise NotImplementedError

    def get_warmup_ratio(self):
        assert self.warmup in ('linear', 'exp')
        alpha = self.last_epoch / self.warmup_iter
        if self.warmup == 'linear':
            ratio = self.warmup_ratio + (1 - self.warmup_ratio) * alpha
        elif self.warmup == 'exp':
            ratio = self.warmup_ratio ** (1. - alpha)
        return ratio
