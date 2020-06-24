import os


class LossInformation(object):
    def update_loss_information(self, train_losses, valid_losses):
        train_losses = train_losses or {}
        valid_losses = valid_losses or {}
        # history
        if not hasattr(self, "_loss_history"):
            self._loss_history = []
            self._train_loss_keys = []
            self._valid_loss_keys = []
        # append new loss
        self._loss_history.append((self.current_epoch, train_losses, valid_losses))
        # update keys
        for key in train_losses:
            if key not in self._train_loss_keys:
                self._train_loss_keys.append(key)
        for key in valid_losses:
            if key not in self._valid_loss_keys:
                self._valid_loss_keys.append(key)
        # dump
        self.__dump_loss_information()

    def __dump_loss_information(self):
        dump_path = os.path.join(self.loss_dir, "epoch-loss.csv")
        meta = (["epoch"]
                + ["[train]" + x for x in self._train_loss_keys]
                + ["[valid]" + x for x in self._valid_loss_keys])
        with open(dump_path, "w") as fp:
            fp.write("{}\n".format(",".join(meta)))
            for (epoch, train_losses, valid_losses) in self._loss_history:
                row = [str(epoch)]
                for pkey in meta[1:]:
                    phase = pkey[:7]
                    key = pkey[7:]
                    if phase == "[train]":
                        row.append(str(train_losses.get(key, "")))
                    else:
                        row.append(str(valid_losses.get(key, "")))
                fp.write("{}\n".format(",".join(row)))
