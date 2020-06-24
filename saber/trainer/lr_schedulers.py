import torch
import numpy as np
from ..utils import log


class _LRScheduler(object):
    def __init__(self, optim, mode, last_step=-1, last_epoch=-1):
        assert mode in ["step", "epoch"]
        assert isinstance(optim, torch.optim.Optimizer),\
            "{} is not an torch.optim.Optimizer".format(type(optim).__name__)
        self.mode = mode
        self.optim = optim
        # set initial_lr
        if last_step == -1 and last_epoch == -1:
            for group in optim.param_groups:
                group.setdefault('initial_lr', group['lr'])
        else:
            for i, group in enumerate(optim.param_groups):
                if 'initial_lr' not in group:
                    group.setdefault('initial_lr', group['lr'])
                    log.warn(
                        "param 'initial_lr' is not specified "
                        "in param_groups[{}] when resuming an optim".format(i)
                    )
        self._last_step = last_step
        self._last_epoch = last_epoch
        self.base_lrs = list(map(
            lambda group: group['initial_lr'],
            optim.param_groups
        ))

    def state_dict(self):
        """Returns the state of the scheduler as a :class:`dict`.

        It contains an entry for every variable in self.__dict__ which
        is not the optim.
        """
        return {
            key: value
            for key, value in self.__dict__.items()
            if key != 'optim'
        }

    def load_state_dict(self, state_dict):
        """Loads the schedulers state.

        Arguments:
            state_dict (dict): scheduler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        self.__dict__.update(state_dict)

    def get_lrs(self):
        raise NotImplementedError

    def step(self, step=None):
        # guard with mode
        if self.mode != "step":
            return
        # default step
        if step is None:
            step = self._last_step + 1
        self._last_step = step
        # update lrs
        for param_group, lr in zip(self.optim.param_groups, self.get_lrs()):
            param_group['lr'] = lr

    def epoch(self, epoch=None):
        # guard with mode
        if self.mode != "epoch":
            return
        # default epoch
        if epoch is None:
            epoch = self._last_epoch + 1
        self._last_epoch = epoch
        # update lrs
        for param_group, lr in zip(self.optim.param_groups, self.get_lrs()):
            param_group['lr'] = lr

    @property
    def last_iter(self):
        if self.mode == "step":
            return self._last_step
        elif self.mode == "epoch":
            return self._last_epoch


class Constant(_LRScheduler):
    def __init__(self, optim, mode="epoch", last_step=-1, last_epoch=-1):
        super(Constant, self).__init__(optim, mode, last_step, last_epoch)

    def get_lrs(self):
        return self.base_lrs


class ExpDecay(_LRScheduler):
    """ Extend default ExpoentialLR with a start_epoch """
    def __init__(self, optim, mode, gamma, last_step=-1, last_epoch=-1,
                 start_iter=50000, gap_iters=1, min_scale=0.001):
        super(ExpDecay, self).__init__(optim, mode, last_step, last_epoch)
        self.gamma = gamma
        self.min_scale = min_scale
        self.start_iter = start_iter
        self.gap_iters = gap_iters

    def get_lrs(self):
        expon = (self.last_iter - self.start_iter) // self.gap_iters
        scale = self.gamma ** max(expon, 0.0)
        scale = max(scale, self.min_scale)
        return [base_lr * scale for base_lr in self.base_lrs]


class NoamDecay(_LRScheduler):
    """ Noam learning rate decay with given warmup step """
    def __init__(self, optim, mode, warmup_iters, last_step=-1, last_epoch=-1):
        super(NoamDecay, self).__init__(optim, mode, last_step, last_epoch)
        self.warmup = warmup_iters

    def get_lrs(self):
        warm_iter = float(self.warmup)
        curr_iter = np.maximum(self.last_iter, 0) + 1
        scale = (
            (warm_iter ** 0.5) *
            np.minimum(curr_iter * (warm_iter ** -1.5),
                       curr_iter ** -0.5)
        )
        return [base_lr * scale for base_lr in self.base_lrs]


class NoamZero(_LRScheduler):
    """ Noam learning rate decay with given warmup step """
    def __init__(self, optim, mode, warmup_iters,
                 start_ramp, total_iters,
                 last_step=-1, last_epoch=-1):
        assert warmup_iters < start_ramp < total_iters
        super(NoamZero, self).__init__(optim, mode, last_step, last_epoch)
        self.warmup = warmup_iters
        self.rzero = start_ramp
        self.total = total_iters
        # ramp beta for adam
        self.betas = None
        for group in optim.param_groups:
            if "betas" in group:
                self.betas = group["betas"]

    def get_lrs(self):
        scale = self._get_scale()
        return [scale * base_lr for base_lr in self.base_lrs]

    def _get_scale(self):
        warm_iter = float(self.warmup)
        curr_iter = np.maximum(self.last_iter, 0) + 1
        scale = (
            (warm_iter ** 0.5) *
            np.minimum(curr_iter * (warm_iter**-1.5), curr_iter**-0.5)
        )
        if curr_iter < self.rzero:
            # noam
            # get betas from groups
            for group in self.optim.param_groups:
                if "betas" in group:
                    group["betas"] = self.betas
        else:
            # ramp to zero
            ramp = float(self.total - curr_iter) / float(self.total - self.rzero)
            ramp = max(min(ramp, 1), 0)
            scale *= ramp
            # ramp beta for adam
            if self.betas is not None:
                new_betas = (
                    self.betas[0] * ramp + 0.5 * (1-ramp),
                    self.betas[1]
                )
                for group in self.optim.param_groups:
                    if "betas" in group:
                        group["betas"] = new_betas
        return scale
