import copy
import errno
import inspect
import numpy as np
import os
import sys

import torch

import pdb


class LoggerOutput(object):
    def __init__(self, fpath=None):
        self.console = sys.stdout
        self.file = None
        if fpath is not None:
            self.mkdir_if_missing(os.path.dirname(fpath))
            self.file = open(fpath, 'w')

    def __del__(self):
        self.close()

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.close()

    def write(self, msg):
        self.console.write(msg)
        if self.file is not None:
            self.file.write(msg)

    def flush(self):
        self.console.flush()
        if self.file is not None:
            self.file.flush()
            os.fsync(self.file.fileno())

    def close(self):
        self.console.close()
        if self.file is not None:
            self.file.close()

    def mkdir_if_missing(self, dir_path):
        try:
            os.makedirs(dir_path)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.initialized = False
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None

    def initialize(self, val, weight):
        self.val = val
        self.avg = val
        self.sum = val*weight
        self.count = weight
        self.initialized = True

    def update(self, val, weight=1):
        val = np.asarray(val)
        if not self.initialized:
            self.initialize(val, weight)
        else:
            self.add(val, weight)

    def add(self, val, weight):
        self.val = val
        self.sum += val * weight
        self.count += weight
        self.avg = self.sum / self.count

    def value(self):
        if self.val is None:
            return 0.
        else:
            return self.val.tolist()

    def average(self):
        if self.avg is None:
            return 0.
        else:
            return self.avg.tolist()


class Struct:
  def __init__(self, *dicts, **fields):
    for d in dicts:
      for k, v in d.iteritems():
        setattr(self, k, v)
    self.__dict__.update(fields)

  def to_dict(self):
    return {a: getattr(self, a) for a in self.attrs()}

  def attrs(self):
    #return sorted(set(dir(self)) - set(dir(Struct)))
    xs = set(dir(self)) - set(dir(Struct))
    xs = [x for x in xs if ((not (hasattr(self.__class__, x) and isinstance(getattr(self.__class__, x), property))) \
        and (not inspect.ismethod(getattr(self, x))))]
    return sorted(xs)

  def updated(self, other_struct_=None, **kwargs):
    s = copy.deepcopy(self)
    if other_struct_ is not None:
      s.__dict__.update(other_struct_.to_dict())
    s.__dict__.update(kwargs)
    return s

  def copy(self):
    return copy.deepcopy(self)

  def __str__(self):
    attrs = ', '.join('%s=%s' % (a, getattr(self, a)) for a in self.attrs())
    return 'Struct(%s)' % attrs


class Params(Struct):
  def __init__(self, **kwargs):
    self.__dict__.update(kwargs)


def normalize_rms(samples, desired_rms=0.1, eps=1e-4):
  rms = torch.max(torch.tensor(eps), torch.sqrt(
      torch.mean(samples**2, dim=1)).float())
  samples = samples * desired_rms / rms.unsqueeze(1)
  return samples


def normalize_rms_np(samples, desired_rms=0.1, eps=1e-4):
  rms = np.maximum(eps, np.sqrt(np.mean(samples**2, 1)))
  samples = samples * (desired_rms / rms)
  return samples


def angle(real, imag): 
  return torch.atan2(imag, real)


def atleast_2d_col(x):
  x = np.asarray(x)
  if np.ndim(x) == 0:
    return x[np.newaxis, np.newaxis]
  if np.ndim(x) == 1:
    return x[:, np.newaxis]
  else:
    return x
