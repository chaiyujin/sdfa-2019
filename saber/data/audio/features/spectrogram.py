import torch
import librosa
import numpy as np
from tqdm import tqdm
from librosa.util import tiny
from librosa.filters import window_sumsquare
from . import misc


class TorchImpl:
    EPSILON = torch.finfo(torch.float).eps

    @staticmethod
    def spectrogram(
        signal, sr,
        win_size,
        hop_size,
        win_fn="hamm",
        padding=False,
        ref_db=20,
        top_db=100,
        normalize=False,
        clip_normalized=True,
        subtract_mean=False,
    ):
        signal, squeeze, is_tensor = TorchImpl.__prepare_batch(signal, target_dims=2)
        fft = torch.stft(
            signal,
            n_fft      = win_size,
            hop_length = hop_size,
            win_length = win_size,
            window     = misc.get_window(win_fn, win_size, signal.device),
            center     = padding,
            pad_mode   = 'constant',
            normalized = False,
            onesided   = True
        )
        linear = fft.pow(2).sum(-1)
        linear = TorchImpl.__power_to_db(linear)
        if normalize:
            linear = TorchImpl.__normalize_db(linear, ref_db, top_db, clip_normalized)
        if subtract_mean:
            linear = TorchImpl.__subtract_mean(linear)
        return TorchImpl.__maybe_squeeze(linear, squeeze, is_tensor)

    @staticmethod
    def inv_spectrogram(
        spec, sr,
        win_size,
        hop_size,
        win_fn="hamm",
        ref_db=20,
        top_db=100,
        normalize=False,
        n_iter=50,
        verbose=False
    ):
        linear, squeeze, is_tensor = TorchImpl.__prepare_batch(spec, target_dims=3)
        if normalize:
            linear = TorchImpl.__denormalize_db(linear, ref_db, top_db)
        power = TorchImpl.__db_to_power(linear)
        amp = torch.sqrt(power)
        wav = TorchImpl.__griffin_lim(amp, win_size, hop_size, win_fn, n_iter, verbose)
        return TorchImpl.__maybe_squeeze(wav, squeeze, is_tensor)

    @staticmethod
    def mel_spectrogram(
        signal, sr,
        win_size,
        hop_size,
        win_fn="hamm",
        padding=False,
        n_mels=80,
        fmin=25,
        fmax=7600,
        ref_db=20,
        top_db=100,
        normalize=False,
        clip_normalized=True,
        subtract_mean=False,
    ):
        signal, squeeze, is_tensor = TorchImpl.__prepare_batch(signal, target_dims=2)
        mel_filters = misc.get_mel_filters(sr, win_size, n_mels, fmin, fmax, signal.device)
        window = misc.get_window(win_fn, win_size, signal.device)
        fft = torch.stft(
            signal,
            n_fft      = win_size,
            hop_length = hop_size,
            win_length = win_size,
            window     = window,
            center     = padding,
            pad_mode   = 'constant',
            normalized = False,
            onesided   = True
        )
        linear = fft.pow(2).sum(-1)
        mel = torch.matmul(mel_filters, linear)
        mel = TorchImpl.__power_to_db(mel)
        if normalize:
            mel = TorchImpl.__normalize_db(mel, ref_db, top_db, clip_normalized)
        if subtract_mean:
            mel = TorchImpl.__subtract_mean(mel)
        return TorchImpl.__maybe_squeeze(mel, squeeze, is_tensor)

    @staticmethod
    def inv_mel_spectrogram(
        spec, sr,
        win_size,
        hop_size,
        win_fn="hamm",
        n_mels=80,
        fmin=25,
        fmax=7600,
        ref_db=20,
        top_db=100,
        normalize=False,
        n_iter=50,
        verbose=False
    ):
        mel, squeeze, is_tensor = TorchImpl.__prepare_batch(spec, target_dims=3)
        inv_mel_filters = misc.get_inv_mel_filters(sr, win_size, n_mels, fmin, fmax, mel.device)
        if normalize:
            mel = TorchImpl.__denormalize_db(mel, ref_db, top_db)
        power = TorchImpl.__db_to_power(mel)
        power = torch.matmul(inv_mel_filters, power)
        amp = torch.sqrt(torch.clamp(power, min=1e-10))

        wav = TorchImpl.__griffin_lim(amp, win_size, hop_size, win_fn, n_iter, verbose)
        return TorchImpl.__maybe_squeeze(wav, squeeze, is_tensor)

    @staticmethod
    def __griffin_lim(spectrogram, win_size, hop_size, win_fn, n_iter=20, verbose=False):
        window = misc.get_window(win_fn, win_size, spectrogram.device)

        def stft(x):
            return torch.stft(
                x,
                n_fft      = win_size,
                hop_length = hop_size,
                win_length = win_size,
                window     = window,
                center     = True,
                pad_mode   = 'constant',
                normalized = False,
                onesided   = True
            )

        def istft(stft):
            B, L, N, _ = stft.size()
            y = stft.new_zeros((B, win_size + hop_size * (N - 1)))
            # generate mask to conjugate
            mask = stft.new_ones((1, win_size, 2))
            mask[:, L:, 1] *= -1
            idx = list(range(L)) + list(range(L-2, 0, -1))
            idx = torch.LongTensor(idx).to(stft.device)
            # batch
            batch_spec = stft.permute(0, 2, 1, 3).contiguous().view(B*N, L, 2)
            batch_spec = batch_spec.index_select(1, idx) * mask
            batch_ytmp = torch.ifft(batch_spec, 1)[..., 0] * window
            batch_ytmp = batch_ytmp.view(B, N, win_size)
            sample = 0
            for i in range(N):
                y[:, sample:(sample+win_size)] += batch_ytmp[:, i, :]
                sample += hop_size
            window_sum = window_sumsquare(
                "hann", N,
                n_fft=win_size, win_length=win_size,
                hop_length=hop_size, dtype=np.float32
            )
            approx_nonzero_indices = torch.from_numpy(
                np.where(window_sum > tiny(window_sum))[0]
            ).to(stft.device)
            window_sum = torch.from_numpy(window_sum).to(stft.device)
            y[:, approx_nonzero_indices] /= window_sum[approx_nonzero_indices]
            return y[:, win_size//2:-win_size//2]

        def to_complex(t):
            ct = t.new_zeros(t.size() + (2,))
            ct[..., 0] = t
            return ct

        def exp_of_scale_imag(t_real, scale_imag):
            t = t_real * scale_imag
            comp = t_real.new_zeros(t.size() + (2,))
            comp[..., 0] = torch.cos(t)
            comp[..., 1] = torch.sin(t)
            return comp

        def mul_complex(a, b):
            c = a.new_zeros(a.size())
            c[..., 0] = a[..., 0] * b[..., 0] - a[..., 1] * b[..., 1]
            c[..., 1] = a[..., 0] * b[..., 1] + a[..., 1] * b[..., 0]
            return c

        spec = spectrogram
        angles = np.pi * np.random.rand(*spec.size()).astype(np.float32)
        angles = torch.from_numpy(angles).to(spec.device)
        angles = exp_of_scale_imag(angles, 2)
        spec_comp = to_complex(torch.abs(spec))
        t = tqdm(range(n_iter), disable=not verbose)
        for i in t:
            full = mul_complex(spec_comp, angles)
            inverse = istft(full)
            rebuilt = stft(inverse)
            angles = torch.atan2(rebuilt[..., 1], rebuilt[..., 0])
            angles = exp_of_scale_imag(angles, 1)
        full = mul_complex(spec_comp, angles)
        inverse = istft(full)
        return inverse

    @staticmethod
    def __prepare_batch(signal, target_dims):
        squeeze = False
        is_tensor = torch.is_tensor(signal)
        if not is_tensor:
            # auto convert to tensor
            signal = torch.FloatTensor(signal)
            # # auto cuda
            # if torch.cuda.is_available():
            #     signal = signal.cuda()
        if signal.dim() == target_dims - 1:
            squeeze = True
            signal = signal.unsqueeze(0)
        assert signal.dim() == target_dims
        return signal, squeeze, is_tensor

    @staticmethod
    def __maybe_squeeze(ret, squeeze, is_tensor):
        if squeeze:
            ret.squeeze_(0)
        if not is_tensor:
            ret = ret.detach().cpu().numpy()
        return ret

    @staticmethod
    def __power_to_db(power):
        return 10.0 * torch.log10(torch.clamp(power, min=TorchImpl.EPSILON))

    @staticmethod
    def __db_to_power(db):
        return torch.pow(10.0, 0.1 * db)

    @staticmethod
    def __normalize_db(db, ref_db, top_db, clip):
        db = (db - ref_db + top_db) / top_db
        if clip:
            db = torch.clamp(db, 0.0, 1.0)
        return db

    @staticmethod
    def __denormalize_db(norm_db, ref_db, top_db):
        return norm_db * top_db - top_db + ref_db

    @staticmethod
    def __subtract_mean(values):
        means = torch.mean(values, dim=-1).unsqueeze(-1)
        values = values - means
        return values


__backend = TorchImpl


def set_backend(backend):
    global __backend
    if backend == "torch":
        __backend = TorchImpl
    else:
        __backend = None


def spectrogram(
    signal, sr, win_size, hop_size, win_fn="hamm", padding=False,
    ref_db=20, top_db=100, normalize=False, clip_normalized=True,
    subtract_mean=False, preemphasis=0
):
    signal = misc.preemphasis(signal, preemphasis)
    return __backend.spectrogram(
        signal, sr, win_size, hop_size, win_fn, padding,
        ref_db, top_db, normalize, clip_normalized,
        subtract_mean
    )


def inv_spectrogram(
    spec, sr, win_size, hop_size, win_fn="hamm",
    ref_db=20, top_db=100, normalize=False,
    n_iter=50, verbose=False, preemphasis=0
):
    inv = __backend.inv_spectrogram(
        spec, sr, win_size, hop_size, win_fn,
        ref_db, top_db, normalize, n_iter, verbose
    )
    return misc.deemphasis(inv, preemphasis)


def mel_spectrogram(
    signal, sr, win_size, hop_size, win_fn="hamm", padding=False,
    n_mels=80, fmin=25, fmax=7600, ref_db=20, top_db=100, normalize=False,
    clip_normalized=True, subtract_mean=False, preemphasis=0
):
    signal = misc.preemphasis(signal, preemphasis)
    return __backend.mel_spectrogram(
        signal, sr, win_size, hop_size, win_fn, padding,
        n_mels, fmin, fmax, ref_db, top_db, normalize,
        clip_normalized, subtract_mean
    )


def inv_mel_spectrogram(
    spec, sr, win_size, hop_size, win_fn="hamm",
    n_mels=80, fmin=25, fmax=7600, ref_db=20, top_db=100,
    normalize=False, n_iter=50, verbose=False, preemphasis=0
):
    inv = __backend.inv_mel_spectrogram(
        spec, sr, win_size, hop_size, win_fn,
        n_mels, fmin, fmax, ref_db, top_db,
        normalize, n_iter, verbose
    )
    return misc.deemphasis(inv, preemphasis)
