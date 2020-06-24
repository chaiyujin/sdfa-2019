import math
import numpy as np
from tqdm import tqdm


class BilateralFilter1D(object):
    '''
        distance_sigma: distance sigma,
        rs: range sigma
        radius: half length of gaussian kernel
    '''
    def __init__(self, factor=-0.5, distance_sigma=1.0, range_sigma=1.0, radius=5):
        self._factor = float(factor)
        self._ds = float(distance_sigma)
        self._rs = float(range_sigma)
        self._radius = int(radius)
        self.__build_distance_weight_table()
        self.__build_similarity_weight_table()

    def __build_distance_weight_table(self):
        self._distance_w = []
        for idx in range(-self._radius, self._radius + 1):
            delta = (float(idx) / self._ds)
            self._distance_w.append(
                math.exp(delta * delta * self._factor))

    def __build_similarity_weight_table(self):
        pass

    def distance_weight(self, dis):
        return self._distance_w[dis + self._radius]

    def similarity_weight(self, delta):
        delta = math.sqrt(delta * delta) / self._rs
        return math.exp(delta * delta * self._factor)

    def __call__(self, signal, verbose=False):
        signal = np.asarray(signal)
        shape = signal.shape
        dtype = signal.dtype
        # reshape the signal
        if len(signal.shape) > 2:
            w = 1
            for i in range(1, len(signal.shape)):
                w *= signal.shape[i]
            signal = np.reshape(signal, (len(signal), w))
        # expand if needed
        if len(signal.shape) == 1:
            signal = np.reshape(signal, (len(signal), 1))
        # filter
        r = self._radius
        new_sig = []
        progress = range(len(signal))
        if verbose:
            progress = tqdm(progress, desc='filtering', leave=False)
        for idx in progress:
            tmp = []
            for dim in range(len(signal[idx])):
                mean = 0.0
                ws = 0.0
                for di in range(-r, r + 1):
                    ni = di + idx
                    if ni < 0 or ni >= len(signal):
                        continue
                    dw = self.distance_weight(di)
                    sw = self.similarity_weight(
                        signal[idx][dim] - signal[ni][dim])
                    weight = dw * sw
                    ws += weight
                    mean += weight * signal[ni][dim]
                mean /= ws
                tmp.append(mean)
            new_sig.append(tmp)
        return np.reshape(new_sig, shape).astype(dtype)
