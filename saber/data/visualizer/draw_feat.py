import torch
import numpy as np


def draw_audio_feature(feat, vmin=None, vmax=None) -> np.ndarray:
    from ... import visualizer
    import matplotlib.pyplot as plt
    if torch.is_tensor(feat):
        feat = feat.detach().cpu().numpy()
    if feat.ndim == 1:
        fig = plt.figure(figsize=(6, 3))
        ax = fig.add_subplot(111)
        ax.tick_params(labelsize=12)
        ax.get_xaxis().set_visible(False)
        plt.plot(feat)
        plt.tight_layout()
        img = visualizer.figure_to_numpy(fig)
        plt.close(fig)
    elif feat.ndim == 2:
        img = visualizer.color_mapping(feat, vmin, vmax, flip_rows=True)
    else:
        raise NotImplementedError("Cannot draw {}d feature.".format(feat.ndim))
    return img
