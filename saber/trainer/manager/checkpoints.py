import os
import re
import torch
from saber.utils import log
from shutil import copyfile


class CheckpointIO(object):

    def load_checkpoint(self, load_from, load_optim=True):
        assert os.path.exists(load_from), f"Failed to find checkpoint: {load_from}"

        # load onto cpu first
        ckpt = torch.load(load_from, map_location='cpu')
        start_epoch = ckpt["epoch"]
        global_step = ckpt.get("global_step")
        if global_step is None:
            global_step = ckpt.get("step", 0)
        log.info(f"load from {start_epoch} epoch ({global_step} step, from '{load_from}').")

        # load model
        try:
            self.model.load_state_dict(ckpt["state"])
        except RuntimeError as err:
            # partially load
            if self.training:
                log.warn("Failed to load model, try to load partially")
                self.model.load_state_dict(ckpt["state"], strict=False)
            else:
                log.fatal("failed to load model, {}", err)

        # load optim
        if load_optim and self.training:
            for name in self.optimizers:
                optim, scheduler = self.optimizers[name]
                try:
                    optim.load_state_dict(ckpt["optim_{}".format(name)])
                    if scheduler is not None:
                        scheduler.load_state_dict(ckpt["lr_scheduler_{}".format(name)])
                except Exception:
                    log.warn("Failed to load optim, ingore.")

        # set indices
        self.model.current_epoch = start_epoch
        self.model.global_step = global_step

    def dump_checkpoint(self, filename=None, max_nb=10, extra_info=""):
        m = self.model.module if self.is_parallel else self.model
        cp = dict(
            epoch       = m.current_epoch,
            global_step = m.global_step
        )

        # model state
        cp["state"]  = m.state_dict()

        # optim and lr_schedulers
        for name in self.optimizers:
            optim, scheduler = self.optimizers[name]
            cp["optim_{}".format(name)] = optim.state_dict()
            cp["lr_scheduler_{}".format(name)] = scheduler.state_dict() if scheduler is not None else None

        # saving
        if filename is None:
            save_path = os.path.join(self.checkpoint_dir, CheckpointIO.__file_at_step(m.current_epoch, m.global_step))
            last_path = os.path.join(self.checkpoint_dir, CheckpointIO.__file_last())
            torch.save(cp, save_path)
            copyfile(save_path, last_path)

            # delete old checkpoints beyond max checkpoints
            if max_nb is not None and max_nb > 0:
                history = []
                for file_name in os.listdir(self.checkpoint_dir):
                    match = re.match(r"^epoch(\d+)-step(\d+)\.ckpt$", file_name)
                    if match is not None:
                        _epoch = int(match.group(1))
                        _step = int(match.group(2))
                        history.append((_epoch, _step))
                history.sort(key=lambda tup: tup[1])
                while len(history) > max_nb:
                    path = os.path.join(self.checkpoint_dir, CheckpointIO.__file_at_step(*history[0]))
                    if os.path.exists(path):
                        os.remove(path)
                        log.info("remove {} to keep {} checkpoints".format(path, max_nb))
                    history.pop(0)
                log.info("current checkpoints at {} steps".format(history))
        else:
            save_path = os.path.join(self.checkpoint_dir, filename)
            info_path = os.path.splitext(save_path)[0] + ".info"
            torch.save(cp, save_path)
            with open(info_path, "w") as fp:
                fp.write("epoch: {}, global_step: {}, {}".format(
                    cp["epoch"], cp["global_step"], extra_info
                ))

    @staticmethod
    def __file_at_step(epoch, step):
        return "epoch{}-step{}.ckpt".format(
            str(epoch).zfill(4),
            str(step).zfill(6)
        )

    @staticmethod
    def __file_last():
        return "last.ckpt"
