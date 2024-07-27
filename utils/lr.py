import torch
import warnings
from torch.optim.lr_scheduler import _LRScheduler

class CustomExponentialLR(_LRScheduler):
    def __init__(self, optimizer, decay_rates, last_epoch=-1, verbose=False):
        # decay_rates should be a list of decay rates for each parameter group
        if len(optimizer.param_groups) != len(decay_rates):
            raise ValueError("Decay rates list should match the number of param groups")

        self.decay_rates = decay_rates
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                          "please use `get_last_lr()`.", UserWarning)

        return [group['lr'] * decay_rate
                for group, decay_rate in zip(self.optimizer.param_groups, self.decay_rates)]

"""
class CustomGroupLR(_LRScheduler):
    def __init__(self, optimizer, decay_rates, intervals, last_epoch=-1, verbose=False):
        # Ensure that decay rates and intervals match the number of parameter groups
        if not len(optimizer.param_groups) == len(decay_rates) == len(intervals):
            raise ValueError("Decay rates and intervals lists must match the number of parameter groups")

        self.decay_rates = decay_rates
        self.intervals = intervals
        self.counters = [0] * len(optimizer.param_groups)  # Initialize counters for each group
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        new_lrs = []
        for i, (group, decay_rate) in enumerate(zip(self.optimizer.param_groups, self.decay_rates)):
            if self.counters[i] % self.intervals[i] == 0:
                new_lrs.append(group['lr'] * decay_rate)
            else:
                new_lrs.append(group['lr'])
            self.counters[i] += 1
        return new_lrs


"""

class CustomGroupLR(_LRScheduler):
    def __init__(self, optimizer, phase_settings, last_epoch=-1, verbose=False):
        self.phase_settings = phase_settings  # List of (num_iters, lr_rates) tuples
        self.current_phase = 0
        self.iter_in_phase = 0
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        # Check if current phase needs to be updated
        if self.iter_in_phase >= self.phase_settings[self.current_phase][0]:
            self.current_phase = (self.current_phase + 1) % len(self.phase_settings)
            self.iter_in_phase = 0
        lr_rates = self.phase_settings[self.current_phase][1]

        new_lrs = []
        for i, param_group in enumerate(self.optimizer.param_groups):
            # Apply the learning rate from the current phase setting
            new_lrs.append(param_group['initial_lr'] * lr_rates[i])
        self.iter_in_phase += 1
        return new_lrs