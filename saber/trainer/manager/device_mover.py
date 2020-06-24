import torch
import saber
from saber.utils import log
from collections import defaultdict
from ..saber_model import SaberDataParallel


class DeviceMover(object):

    def move_to(self, device):
        if device != "cpu" and not torch.cuda.is_available():
            device = "cpu"
            log.warn("cuda is not available!")
        if self.training:
            if device != "cpu":
                # optims
                for name in self.optimizers:
                    DeviceMover._cuda_optim(self.optimizers[name][0])
                self.model._on_gpu = True
                self.model.cuda()
            # multi-gpu
            if isinstance(device, (list, tuple)):
                log.info("DataParallel with {}".format(device))
                self.model = SaberDataParallel(self.model, device_ids=device)
                self.model.cuda()
        else:
            # remove weight norm for fast inference
            saber.nn.functions.remove_weight_norm(self.model)
            if device != "cpu":
                self.model._on_gpu = True
                self.model.cuda()

    @staticmethod
    def _cuda_optim(optim):
        if not hasattr(optim, "state"):
            return

        def move_cuda(D):
            for k in D:
                if isinstance(D[k], dict) or isinstance(D[k], defaultdict):
                    move_cuda(D[k])
                elif torch.is_tensor(D[k]):
                    D[k] = D[k].cuda()

        move_cuda(optim.state)
