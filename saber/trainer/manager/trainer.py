import torch
from enum import Enum
from tqdm import tqdm
from saber.utils import log
from collections import defaultdict
from .experiment import Experiment
from ..saber_model import SaberModel


class Trainer(object):

    class hooks(Enum):
        prev_train = "prev_train"
        post_train = "post_train"
        prev_valid = "prev_valid"
        post_valid = "post_valid"
        prev_epoch = "prev_epoch"
        post_epoch = "post_epoch"

    _hook_dict: dict = {
        hooks.prev_train: [], hooks.post_train: [],
        hooks.prev_valid: [], hooks.post_valid: [],
        hooks.prev_epoch: [], hooks.post_epoch: [],
    }

    @classmethod
    def register(cls, hook_type: hooks):
        assert type(hook_type) is Trainer.hooks

        def wrapper(fn):
            _check_hook_fn(fn, hook_type)
            cls._hook_dict[hook_type].append(fn)
            return fn

        return wrapper

    def __init__(self, experiment: Experiment):
        super().__init__()
        # set hyper parameters
        experiment.hparams.check_keys("trainer")
        self.experiment = experiment
        self.hp = experiment.hparams.trainer

        def _ep_st(prefix):
            return f"Please set either '{prefix}_epochs' or '{prefix}_steps', but not both!"

        # set epochs/steps and check
        self._max_epochs        = self.hp.get("max_epochs")
        self._valid_gap_epochs  = self.hp.get("valid_gap_epochs")
        self._eval_gap_epochs   = self.hp.get("eval_gap_epochs")
        self._save_gap_epochs   = self.hp.get("save_gap_epochs")
        self._max_steps         = self.hp.get("max_steps")
        self._valid_gap_steps   = self.hp.get("valid_gap_steps")
        self._eval_gap_steps    = self.hp.get("eval_gap_steps")
        self._save_gap_steps    = self.hp.get("save_gap_steps")
        assert (self._max_epochs       is not None) ^ (self._max_steps       is not None), f"{_ep_st('max')}"
        assert (self._valid_gap_epochs is not None) ^ (self._valid_gap_steps is not None), f"{_ep_st('valid_gap')}"
        assert (self._eval_gap_epochs  is not None) ^ (self._eval_gap_steps  is not None), f"{_ep_st('eval_gap')}"
        assert (self._save_gap_epochs  is not None) ^ (self._save_gap_steps  is not None), f"{_ep_st('save_gap')}"
        # evaluate at zero (default is False, since model is poor at this moment)
        self._eval_debug        = self.hp.get_default("eval_debug", False)

        # set steps
        self._grad_acc_steps    = self.hp.get("grad_acc_steps", 1)
        self._plot_gap_steps    = self.hp.get("plot_gap_steps", 100)
        self._scalar_gap_steps  = self.hp.get("scalar_gap_steps", 50)

        # set grad clip
        self._grad_clip         = self.hp.get("max_grad_clip", None)
        self._grad_norm         = self.hp.get("max_grad_norm", None)

        # about ckpts
        self._max_ckpts         = self.hp.get("max_checkpoints", 10)
        self._ref_metric        = self.hp.get("reference_metric", None)
        self._ref_metric_larger = self.hp.get("reference_metric_larger", False)
        self._best_metric       = None

        # loss details
        self._loss_detail       = self.hp.get("loss_detail", False)

        # implemented flag
        self._impl_valid_step   = self.experiment.saber_model.has_implemented("valid_step")
        self._impl_eval         = self.experiment.saber_model.has_implemented("evaluate")
        if self._impl_eval:
            self.hp.check_keys("evaluate")

        # cached indices
        self._i_batch = None
        self._current_epoch = None
        self._global_step = None

    # --------------- #
    #       api       #
    # --------------- #

    def train(self):
        # prepare loss dict for last epoch
        self.train_epoch_losses = {}
        self.valid_epoch_losses = {}
        self.train_epoch_scalar = {}
        self.valid_epoch_scalar = {}

        # iter epoch
        max_epochs = self._max_epochs if self._max_epochs is not None else 10000000
        for epoch in range(self.current_epoch, max_epochs):
            # _. debug
            if self._eval_debug and epoch == self.current_epoch:
                self._evaluating()

            # _. increase epoch
            self.current_epoch = epoch + 1

            # _. hooks prev epoch
            for hook in self._hook_dict[Trainer.hooks.prev_epoch]:
                hook(trainer=self)

            # 1. train epoch
            self._train_epoch()

            # 2. maybe valid at epoch
            if self._should_at_epoch(epoch, self._valid_gap_epochs):
                self._validation()

            # 3. maybe eval at epoch
            if self._should_at_epoch(epoch, self._eval_gap_epochs):
                self._evaluating()

            # 4. maybe save checkpoint
            if self._should_at_epoch(epoch, self._save_gap_epochs):
                self.experiment.dump_checkpoint(max_nb=self._max_ckpts)

            # 5. experiment dump loss information
            self.experiment.update_loss_information(
                {**self.train_epoch_losses, **self.train_epoch_scalar},
                {**self.valid_epoch_losses, **self.valid_epoch_scalar}
            )

            # _. hooks after epoch
            for hook in self._hook_dict[Trainer.hooks.post_epoch]:
                hook(trainer=self)

            # terminate by max steps
            if self._max_steps is not None and self.global_step >= self._max_steps:
                break

        # finally dump checkpoint
        self.experiment.dump_checkpoint(max_nb=None)

    def valid(self):
        self._validation()

    # --------------- #
    #  learning rate  #
    # --------------- #

    def epoch_lr_schedulers(self):
        for name, (_, scheduler) in self.experiment.optimizers.items():
            if scheduler is not None:
                scheduler.epoch(self.current_epoch)

    def step_lr_schedulers(self):
        for name, (_, scheduler) in self.experiment.optimizers.items():
            if scheduler is not None:
                scheduler.step(self.global_step)

    def current_lr_dict(self):
        ret = {}
        for name, (optim, _) in self.experiment.optimizers.items():
            lr_list = []
            for param_group in optim.param_groups:
                lr_list.append(param_group['lr'])
            for i, lr in enumerate(lr_list):
                ret["{}-group{}".format(name, i)] = lr
        return ret

    # ---------------- #
    #  grads / update  #
    # ---------------- #

    def step_optimizers(self):
        for name, (optim, _) in self.experiment.optimizers.items():
            optim.step()

    def zero_grad(self):
        self.experiment.model.zero_grad()
        for name, (optim, _) in self.experiment.optimizers.items():
            optim.zero_grad()

    def maybe_clip_grad(self):
        import torch.nn.utils as utils
        if self._grad_clip is not None:
            utils.clip_grad_value_(self.experiment.model.parameters(), self._grad_clip)
        if self._grad_norm is not None:
            utils.clip_grad_norm_(self.experiment.model.parameters(), self._grad_norm)

    # ------------- #
    #  check tools  #
    # ------------- #

    def _should_at_epoch(self, epoch, gap_epochs):
        return (gap_epochs is not None) and (gap_epochs > 0) and (epoch % gap_epochs == 0)

    def _should_at_step(self, step, gap_steps):
        return (gap_steps is not None) and (gap_steps > 0) and (step % gap_steps == 0)

    def _check_step_return(self, phase, ret):
        assert (
            isinstance(ret, tuple)
            and isinstance(ret[0], dict)
            and isinstance(ret[1], dict)
            and (
                (len(ret) == 3 and isinstance(ret[2], dict))
                or (len(ret) == 2)
            )
        ), "{}".format(SaberModel.__step_should_return__.format(phase))
        if phase == "train":
            assert len(ret[1]) > 0, "train_step() return an empty loss_dict!"

    # ---------------------- #
    #  main epoch functions  #
    # ---------------------- #

    def _cuda_batch(self, batch):
        if isinstance(batch, dict):
            for k in batch:
                batch[k] = self._cuda_batch(batch[k])
        elif isinstance(batch, (tuple, list)):
            batch = list(self._cuda_batch(x) for x in batch)
        elif torch.is_tensor(batch):
            batch = batch.cuda()
        return batch

    def _train_epoch(self):
        # check train loaders
        assert len(self.experiment.train_loaders) > 0,\
            "[trainer]: No train loader is given!"

        self.train_epoch_losses = defaultdict(lambda: 0.0)
        self.train_epoch_scalar = defaultdict(lambda: 0.0)
        progress = tqdm(self.experiment.main_train_loader)

        # update lr schedulers by epoch
        self.epoch_lr_schedulers()
        for self._i_batch, batch in enumerate(progress):
            # increase global step
            self.global_step += 1

            # update lr schedulers by step
            self.step_lr_schedulers()
            # make sure is train mode
            self.experiment.model.train()
            # if on gpu, cuda batch
            if self.experiment.on_gpu:
                batch = self._cuda_batch(batch)

            # hook before train step
            for hook in self._hook_dict[Trainer.hooks.prev_train]:
                hook(trainer=self, batch=batch)

            # ----------------------------------- #
            #              main step              #
            # ----------------------------------- #

            # !important: clear grads, must -1
            if self._should_at_step(self.global_step-1, self._grad_acc_steps):
                self.zero_grad()

            with torch.autograd.detect_anomaly():
                # forward phase
                ret_tuple = self.experiment.model.train_step(
                    batch   = batch,
                    i_batch = self._i_batch
                )

                # check return
                self._check_step_return("train", ret_tuple)
                pred_dict = ret_tuple[0]
                loss_dict = ret_tuple[1]
                sclr_dict = ret_tuple[2] if len(ret_tuple) >= 3 else {}

                # backward phase
                loss = _sumup_batch_mean_loss(loss_dict)
                loss.backward()  # accumulate grads

                # !important: using aux train loaders
                for aux_name in self.experiment.train_iterators.keys():
                    assert aux_name != self.experiment.main_loader_name
                    # get next aux batch
                    try:
                        aux_batch = next(self.experiment.train_iterators[aux_name])
                    except StopIteration:
                        self.experiment.train_iterators[aux_name] = iter(self.experiment.train_loaders[aux_name])
                        aux_batch = next(self.experiment.train_iterators[aux_name])
                    # move to gpu
                    if self.experiment.on_gpu:
                        aux_batch = self._cuda_batch(aux_batch)
                    # forward phase
                    aux_ret = self.experiment.model.train_step(
                        batch   = aux_batch,
                        i_batch = self._i_batch
                    )
                    # check return
                    self._check_step_return("train", aux_ret)
                    aux_pred_dict = aux_ret[0]
                    aux_loss_dict = aux_ret[1]
                    aux_sclr_dict = aux_ret[2] if len(ret_tuple) >= 3 else {}
                    # backward phase
                    aux_loss = _sumup_batch_mean_loss(aux_loss_dict)
                    aux_loss.backward()
                    # extend dictionary
                    for key in aux_batch:     batch    [f"aux-{aux_name}-{key}"] = aux_batch[key]
                    for key in aux_pred_dict: pred_dict[f"aux-{aux_name}-{key}"] = aux_pred_dict[key]
                    for key in aux_loss_dict: sclr_dict[f"aux-{aux_name}-{key}"] = float(aux_loss_dict[key].item())
                    for key in aux_sclr_dict: sclr_dict[f"aux-{aux_name}-{key}"] = float(aux_sclr_dict[key])

            # !important: update phase, must be global_step
            if self._should_at_step(self.global_step, self._grad_acc_steps):
                self.maybe_clip_grad()
                self.step_optimizers()

            # acculate information
            self.train_step_losses  = {k: float(loss_dict[k].sum())                              for k in loss_dict}
            self.train_epoch_losses = {k: float(loss_dict[k].sum()) + self.train_epoch_losses[k] for k in loss_dict}
            self.train_epoch_scalar = {k: float(sclr_dict[k])       + self.train_epoch_scalar[k] for k in sclr_dict}

            # ----------------------------------- #
            #              log step               #
            # ----------------------------------- #
            if self._should_at_step(self.global_step, self._plot_gap_steps) or self.global_step == 1:
                self.experiment.plot_nested("0.train", pred_dict, batch)
            if self._should_at_step(self.global_step, self._scalar_gap_steps) or self.global_step == 1:
                self.experiment.scalar("0.train-step/loss", self.train_step_losses)
                self.experiment.scalar("0.train-step/scalar", sclr_dict)
                self.experiment.scalar("0.train-step/misc/lr", self.current_lr_dict())
                self.experiment.scalar("0.train-step/misc/grad_norm",
                                       self.experiment.saber_model.grad_norm_dict(2)["grad_norm_total"])

            progress.set_description("[{}|train{}]".format(self.current_epoch, self.global_step))
            progress.set_postfix(
                {k: "{:.4f}".format(x) for k, x in self.train_step_losses.items()}
                if self._loss_detail else
                {"loss": "{:.4f}".format(float(loss))}
            )

            # hook after train step
            for hook in self._hook_dict[Trainer.hooks.post_train]:
                hook(trainer=self, batch=batch, pred_dict=pred_dict,
                     loss_dict=loss_dict, scalar_dict=sclr_dict)

            # maybe valid/eval/save by step
            if self._should_at_step(self.global_step, self._valid_gap_steps):
                self._validation()
            if self._should_at_step(self.global_step, self._eval_gap_steps):
                self._evaluating()
            if self._should_at_step(self.global_step, self._save_gap_steps):
                self.experiment.dump_checkpoint(max_nb=self._max_ckpts)

            # terminate by max steps
            if self._max_steps is not None and self.global_step >= self._max_steps:
                break

        # ----------------------------------- #
        #              log epoch              #
        # ----------------------------------- #
        def _log_dict(tag, val_dict):
            if len(val_dict) > 0:
                num = float(self._i_batch + 1)
                for k in val_dict:
                    val_dict[k] /= num
                # the epoch is given as step
                self.experiment.scalar("0.train-epoch/{}".format(tag), val_dict, global_step=self.current_epoch)

        _log_dict("loss",   self.train_epoch_losses)
        _log_dict("scalar", self.train_epoch_scalar)

    def _validation(self):
        # check
        if (
            len(self.experiment.valid_loaders) == 0
            or (not self._impl_valid_step)
        ):
            return

        self.valid_epoch_losses = defaultdict(lambda: 0.0)
        self.valid_epoch_scalar = defaultdict(lambda: 0.0)
        progress = tqdm(self.experiment.main_valid_loader)

        for self._i_batch, batch in enumerate(progress):
            # evaluation mode
            self.experiment.model.eval()
            # move to gpu
            if self.experiment.on_gpu:
                batch = self._cuda_batch(batch)

            # hook before valid step
            for hook in self._hook_dict[Trainer.hooks.prev_valid]:
                hook(trainer=self, batch=batch)

            # ----------------------------------- #
            #              main step              #
            # ----------------------------------- #
            # forward phase
            ret_tuple = self.experiment.model.valid_step(
                batch   = batch,
                i_batch = self._i_batch
            )
            # it's possible that not implemented!
            if ret_tuple is None:
                self._impl_valid_step = False
                return
            # check
            self._check_step_return("valid", ret_tuple)
            pred_dict = ret_tuple[0]
            loss_dict = ret_tuple[1]
            sclr_dict = ret_tuple[2] if len(ret_tuple) >= 3 else {}

            # using aux valid loaders
            for aux_name in self.experiment.valid_iterators.keys():
                assert aux_name != self.experiment.main_loader_name
                try:
                    aux_batch = next(self.experiment.valid_iterators[aux_name])
                except StopIteration:
                    self.experiment.valid_iterators[aux_name] = iter(self.experiment.valid_loaders[aux_name])
                    aux_batch = next(self.experiment.valid_iterators[aux_name])
                if self.experiment.on_gpu:
                    aux_batch = self._cuda_batch(aux_batch)
                aux_ret = self.experiment.model.valid_step(
                    batch   = aux_batch,
                    i_batch = self._i_batch
                )
                self._check_step_return("valid", aux_ret)
                aux_pred_dict = aux_ret[0]
                aux_loss_dict = aux_ret[1]
                aux_sclr_dict = aux_ret[2] if len(ret_tuple) >= 3 else {}
                # extend
                for key in aux_batch:     batch    [f"aux-{aux_name}-{key}"] = aux_batch[key]
                for key in aux_pred_dict: pred_dict[f"aux-{aux_name}-{key}"] = aux_pred_dict[key]
                for key in aux_loss_dict: sclr_dict[f"aux-{aux_name}-{key}"] = float(aux_loss_dict[key].item())
                for key in aux_sclr_dict: sclr_dict[f"aux-{aux_name}-{key}"] = float(aux_sclr_dict[key])

            # ----------------------------------- #
            #              log step               #
            # ----------------------------------- #
            self.valid_step_losses  = {k: float(loss_dict[k].sum())                              for k in loss_dict}
            self.valid_epoch_losses = {k: float(loss_dict[k].sum()) + self.valid_epoch_losses[k] for k in loss_dict}
            self.valid_epoch_scalar = {k: float(sclr_dict[k])       + self.valid_epoch_scalar[k] for k in sclr_dict}

            progress.set_description("[{}|valid{}]".format(self.current_epoch, self.global_step))
            progress.set_postfix(
                {k: "{:.4f}".format(x) for k, x in self.valid_step_losses.items()}
                if self._loss_detail else
                {"loss": "{:.4f}".format(float(_sumup_batch_mean_loss(loss_dict).mean().item()))}
            )

            # hook after valid step
            for hook in self._hook_dict[Trainer.hooks.post_valid]:
                hook(trainer=self, batch=batch, pred_dict=pred_dict,
                     loss_dict=loss_dict, scalar_dict=sclr_dict)

        # ----------------------------------- #
        #              log epoch              #
        # ----------------------------------- #
        def _log_dict(tag, val_dict):
            if len(val_dict) > 0:
                num = float(self._i_batch + 1)
                for k in val_dict:
                    val_dict[k] /= num
                self.experiment.scalar("1.valid-epoch/{}".format(tag), val_dict, global_step=self.current_epoch)

        _log_dict("loss", self.valid_epoch_losses)
        _log_dict("scalar", self.valid_epoch_scalar)

        # ----------------------------------- #
        #    update best reference metric     #
        # ----------------------------------- #
        if self._ref_metric is not None:
            scalars = {**self.valid_epoch_losses, **self.valid_epoch_scalar}
            if self._ref_metric not in scalars:
                log.warn("valid_step didn't return reference metric '{}'".format(self._ref_metric))
                self._ref_metric = None
            elif (
                (self._best_metric is None) or
                (scalars[self._ref_metric] >= self._best_metric and self._ref_metric_larger) or
                (scalars[self._ref_metric] <= self._best_metric and not self._ref_metric_larger)
            ):
                self._best_metric = scalars[self._ref_metric]
                log.info("update best reference metric '{}': {:.6f}".format(self._ref_metric, self._best_metric))
                self.experiment.dump_checkpoint(
                    filename="best-{}.ckpt".format(self._ref_metric),
                    extra_info="{}: {}".format(self._ref_metric, self._best_metric)
                )

    def _evaluating(self):
        log.info("evaluating...")
        self.experiment.model.eval()
        self.experiment.saber_model.evaluate(self.hp.evaluate, self.experiment, in_trainer=True)

    # ---------------- #
    #  cached indices  #
    # ---------------- #

    @property
    def current_epoch(self):
        self._current_epoch = self.experiment.current_epoch
        return self._current_epoch

    @current_epoch.setter
    def current_epoch(self, value):
        self._current_epoch = value
        self.experiment.current_epoch = value

    @property
    def global_step(self):
        return self.experiment.global_step

    @global_step.setter
    def global_step(self, value):
        self._global_step = value
        self.experiment.global_step = value


def _sumup_batch_mean_loss(loss_dict):
    start = 0.0
    for k in loss_dict:
        if isinstance(loss_dict[k], dict):
            val = _sumup_batch_mean_loss(loss_dict[k])
        elif torch.is_tensor(loss_dict[k]):
            assert loss_dict[k].dim() <= 1,\
                f"'{k}' in loss_dict is not a scalar or batched vector!'"
            val = loss_dict[k].mean()
        else:
            # ignore non-tensor
            continue
        start = val + start
    return start


def _check_hook_fn(fn, hook_type: Trainer.hooks):
    import inspect
    params = [k for k in inspect.signature(fn).parameters]
    error_msg = "wrong signature for {}".format(hook_type)
    if hook_type in [Trainer.hooks.prev_train, Trainer.hooks.prev_valid]:
        assert params == ["trainer", "batch"], error_msg
    elif hook_type in [Trainer.hooks.post_train, Trainer.hooks.post_valid]:
        assert params == ["trainer", "batch", "pred_dict", "loss_dict", "scalar_dict"], error_msg
    elif hook_type in [Trainer.hooks.prev_epoch, Trainer.hooks.post_epoch]:
        assert params == ["trainer"], error_msg
