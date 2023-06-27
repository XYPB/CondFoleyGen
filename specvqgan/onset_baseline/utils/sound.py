import copy
import numpy as np
import scipy.io.wavfile
import scipy.signal

from . import utils as ut

import pdb

def load_sound(wav_fname):
    rate, samples = scipy.io.wavfile.read(wav_fname)
    times = (1./rate) * np.arange(len(samples))
    return Sound(times, rate, samples)


class Sound:
    def __init__(self, times, rate, samples=None):
        # Allow Sound(samples, sr)
        if samples is None:
            samples = times
            times = None
        if samples.dtype == np.float32:
            samples = samples.astype('float64')

        self.rate = rate
        # self.samples = ut.atleast_2d_col(samples)
        self.samples = samples

        self.length = samples.shape[0]
        if times is None:
            self.times = np.arange(len(self.samples)) / float(self.rate)
        else:
            self.times = times

    def copy(self):
        return copy.deepcopy(self)

    def parts(self):
        return (self.times, self.rate, self.samples)

    def __getslice__(self, *args):
        return Sound(self.times.__getslice__(*args), self.rate,
                    self.samples.__getslice__(*args))

    def duration(self):
        return self.samples.shape[0] / float(self.rate)

    def normalized(self, check=True):
        if self.samples.dtype == np.double:
            assert (not check) or np.max(np.abs(self.samples)) <= 4.
            x = copy.deepcopy(self)
            x.samples = np.clip(x.samples, -1., 1.)
            return x
        else:
            s = copy.deepcopy(self)
            s.samples = np.array(s.samples, 'double') / np.iinfo(s.samples.dtype).max
            s.samples[s.samples < -1] = -1
            s.samples[s.samples > 1] = 1
            return s

    def unnormalized(self, dtype_name='int32'):
        s = self.normalized()
        inf = np.iinfo(np.dtype(dtype_name))
        samples = np.clip(s.samples, -1., 1.)
        samples = inf.max * samples
        samples = np.array(np.clip(samples, inf.min, inf.max), dtype_name)
        s.samples = samples
        return s

    def sample_from_time(self, t, bound=False):
        if bound:
            return min(max(0, int(np.round(t * self.rate))), self.samples.shape[0]-1)
        else:
            return int(np.round(t * self.rate))

    # st = sample_from_time

    def shift_zero(self):
        s = copy.deepcopy(self)
        s.times -= s.times[0]
        return s

    def select_channel(self, c):
        s = copy.deepcopy(self)
        s.samples = s.samples[:, c]
        return s

    def left_pad_silence(self, n):
        if n == 0:
            return self.shift_zero()
        else:
            if np.ndim(self.samples) == 1:
                samples = np.concatenate([[0] * n, self.samples])
            else:
                samples = np.vstack(
                [np.zeros((n, self.samples.shape[1]), self.samples.dtype), self.samples])
        return Sound(None, self.rate, samples)

    def right_pad_silence(self, n):
        if n == 0:
            return self.shift_zero()
        else:
            if np.ndim(self.samples) == 1:
                samples = np.concatenate([self.samples, [0] * n])
            else:
                samples = np.vstack([self.samples, np.zeros(
                (n, self.samples.shape[1]), self.samples.dtype)])
        return Sound(None, self.rate, samples)

    def pad_slice(self, s1, s2):
        assert s1 < self.samples.shape[0] and s2 >= 0
        s = self[max(0, s1): min(s2, self.samples.shape[0])]
        s = s.left_pad_silence(max(0, -s1))
        s = s.right_pad_silence(max(0, s2 - self.samples.shape[0]))
        return s

    def to_mono(self, force_copy= True):
        s = copy.deepcopy(self)
        s.samples = make_mono(s.samples)
        return s

    def slice_time(self, t1, t2):
        return self[self.st(t1): self.st(t2)]

    @property
    def nchannels(self):
        return 1 if np.ndim(self.samples) == 1 else self.samples.shape[1]

    def save(self, fname):
        s = self.unnormalized('int16')
        scipy.io.wavfile.write(fname, s.rate, s.samples.transpose())

    def resampled(self, new_rate, clip= True):
        if new_rate == self.rate:
            return copy.deepcopy(self)
        else:
        #assert self.samples.shape[1] == 1
            return Sound(None, new_rate, self.resample(self.samples, float(new_rate)/self.rate, clip= clip))

    def trim_to_size(self, n):
        return Sound(None, self.rate, self.samples[:n])

    def resample(self, signal, sc, clip = True, num_samples = None):
        n = int(round(signal.shape[0] * sc)) if num_samples is None else num_samples
        r = scipy.signal.resample(signal, n)
    
        if clip:
            r = np.clip(r, -1, 1)
        else: 
            r = r.astype(np.int16)
        return r
