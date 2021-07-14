from gpu import GPU
import math
import numpy as np
from collections import defaultdict
import scipy.stats as ss

MINS_PER_TICK = 1

def extrapolate_estimate(num_gpus, time_estimates):
    '''
    :param num_gpus: number of gpus given
    :param time_estimates: estimated time for num_gpu
    :return: time it would take for isolated num_gpus
    '''
    POWER = 3/4
    if num_gpus == 0:
        return 0
    elif num_gpus in time_estimates:
        return time_estimates[num_gpus]
    elif len(time_estimates) == 1:
        k, v = tuple(time_estimates.items())[0]
        return num_gpus ** POWER * v / k ** POWER
    else: #basic midpoint estimation; practically linear
        return sum(time_estimates.values()) / sum(time_estimates.keys()) * num_gpus

class Job:
    #time estimates are given by the form of {gpu_config (in naive case num_gpus) : estimated_total_ticks}
    #these estimates will be given in minutes

    #however, start_time and all other times will be in ticks, and therefore need MIN_PER_TICK constant

    def __init__(self, id : int, start_time : int, num_iterations : int, user_time_estimates : {int, int},
                 req_gpu : int, true_time_rates : {int, int} = {}, memory_usage : int = 100):

        self.BASE_GPU = 4
        self.id, self.req_gpu, self.iter_total, self.iter_completed = id, req_gpu, num_iterations, 0
        self.gpus, self.time_elapsed, self.start_time, self.finish_time = set(), 0, start_time, -1

        #user estimated time dictionary
        self.user_time_ests = {k : v / MINS_PER_TICK for k,v in user_time_estimates.items()}
        #true_ests is the true estimate for various full job at various reqs, defaulted to the user's estimate
        self.profile_time_ests = self.user_time_ests

        self.true_rates = true_time_rates
        #execution record is [(gpu config, ticks taken, total iters)]
        #current_exec is just the current one for info while it is running
        self.execution_records = defaultdict(lambda : np.array((0,0), dtype=np.int64))

    def __hash__(self):
        return self.id.__hash__()

    def __str__(self):
        if self.finished:
            return f'Job {self.id}, DONE at time {self.finish_time}'
        if self.num_gpus > 0:
            return f'Job {self.id}, RUN – progress {self.progress}'

        return f'Job {self.id}, PAUSE – progress {self.progress}'

    def allocate_gpus(self, GPUs : {GPU}, timestamp : int = 0):
        #TODO: implement a penalty to switching GPUs
        self.gpus = GPUs
        for gpu in GPUs:
            gpu.assign_job(self.id)

    def deallocate_gpus(self):
        #TODO: implement migration penalty/delay
        for gpu in self.gpus:
            gpu.remove_job(self.id)

        self.gpus.clear()

    def _update_profile_estimates(self):
        #update self.profile_time_ests

        #execution records:
        #gpu config : [time completed, iterations done]

        curr_record = self.execution_records[self.gpu_config]

        if self.gpu_config not in self.user_time_ests:
            self.profile_time_ests[self.gpu_config] = curr_record[0] * self.iter_total / curr_record[1]
        else:
            self.profile_time_ests[self.gpu_config] = (curr_record[0]**2 * self.iter_total / curr_record[1] +
                                                       self.user_time_ests[self.gpu_config]) / (curr_record[0] + 1)


    def _perform_iteration(self):
        true_est_time = extrapolate_estimate(self.num_gpus, self.true_rates) / MINS_PER_TICK
        estimated_iters_per_tick = math.floor(self.iter_total / true_est_time)

        noise = ss.norm.rvs(0, .1)
        iters = estimated_iters_per_tick * (1+noise) #get the numbers of iters, with noise

        self.iter_completed += iters


    def tick(self):
        self.time_elapsed += 1
        if self.num_gpus == 0:
            return

        prev_iters = self.iter_completed
        self._perform_iteration()
        self.execution_records[self.gpu_config] += np.array((1, self.iter_completed - prev_iters), dtype=np.int64)

        self._update_profile_estimates()

        if self.finished:
            self.deallocate_gpus()
            self.finish_time = self.start_time + self.time_elapsed
            self.execution_records = dict(self.execution_records)

    @property
    def finished(self):
        return self.progress >= 1

    @property
    def allocated_gpus(self):
        return self.gpus

    @property
    def num_gpus(self):
        return len(self.gpus)

    @property
    def progress(self):
        return self.iter_completed / self.iter_total

    @property
    def est_finish_time(self):
        return extrapolate_estimate(self.BASE_GPU, self.profile_time_ests) + self.start_time

    @property
    def gpu_config(self):
        return self.num_gpus