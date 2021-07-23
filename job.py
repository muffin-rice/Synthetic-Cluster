import numpy as np
import scipy.stats as ss

SEC_PER_TICK = 1
UPDATE_FREQ = 10

#bigger models tend to have worse inter-server perofrmance

def extrapolate_estimate(num_gpus, time_estimates : {}):
    '''
    :param num_gpus: number of gpus given
    :param time_estimates: estimated time for num_gpu
    :return: time it would take for isolated num_gpus
    "serial" as coined in THEMIS
    '''
    POWER = 5/6
    if num_gpus == 0:
        return 0
    elif num_gpus in time_estimates:
        return time_estimates[num_gpus]
    elif len(time_estimates) == 1:
        k, v = tuple(time_estimates.items())[0]
        return num_gpus ** POWER * v / k ** POWER
    else: #basic midpoint estimation; practically linear
        return sum(time_estimates.values()) / sum(time_estimates.keys()) * num_gpus

def estimate_server_factor(num_machines : int, model_size : int):
    #num machines = 1 -> factor = 1
    asymptote = min(0.8, 5 / model_size ** .5)
    factor = asymptote + (1-asymptote) / 2 ** ((num_machines-1)/2)

    return factor

class Job:
    #the actual job; separated from the profiled job

    def __init__(self, id : int, start_time : int, total_iterations : int, pref_gpus : (int, int),
                 true_times : dict, memory_usage : int, model_size : int):
        '''

        :param id: ID of GPU
        :param start_time: start time of job
        :param total_iterations: total number of iterations
        :param pref_gpus: user-inputted list of range of gpus they want, inclusive (if only 1, pref_gpu[0] = pref_gpu[1])
        :param true_times: time it takes for completion
        :param memory_usage: memory imprint on a GPU
        :param model_size:
        '''
        self.id, self.start_time, self.iter_total, self.pref_gpus, self.true_times, self.memory_usage, self.model_size =\
        id, start_time, total_iterations, pref_gpus, true_times, memory_usage, model_size

        self.iter_completed = 0
        #{Machine : [GPU]}
        #the idea is that interserver interference (machines) and intraserver interference (gpus) are independent
        self.gpu_config = {}
        self.time_elapsed = 0
        self.time_to_finish = 0
        self.time_until_free = 0

    def __hash__(self):
        x = f'Job{self.id}'
        return x.__hash__()

    def __str__(self):
        if self.finished:
            return f'Job {self.id}, DONE at time {self.finish_time}'
        if self.num_gpus > 0:
            return f'Job {self.id}, RUN – progress {self.progress}'

        return f'Job {self.id}, PAUSE – progress {self.progress}'

    def __repr__(self): 
        return self.__str__()

    def _get_factors(self):
        '''computes the factor of slowdown from the various factors (while training)'''

        #packing factor interference
        if self.num_gpus == 1:
            return 1

        #the "true" factor is the factor that's the slowest (no additional packing factor)
        intraserver_locality = np.min([mach.get_machine_locality_factor(self) for mach in self.gpu_config.keys()])
        #network factor
        interserver_locality = estimate_server_factor(self.num_gpus, self.model_size)
        noise = ss.norm.rvs(0, .1)

        total_change = intraserver_locality * interserver_locality * (1+noise)

        return total_change

    def _perform_iteration(self):
        '''computes #iterations/tick, includes the factors as well'''
        interference = self._get_factors()
        #estimated time to finish the whole job
        estimated_time = extrapolate_estimate(self.num_gpus, self.true_times)
        iters_per_tick = self.iter_total / estimated_time
        adjusted_iters = int(iters_per_tick * interference)

        self.iter_completed += adjusted_iters

    def tick(self):
        if self.finished: #should ideally never be true (job should be in finished_jobs in cluster and never tick)
            return

        self.time_elapsed += 1

        if self.time_until_free > 0:
            self.time_until_free -= 1

        if self.num_gpus == 0:
            return

        self._perform_iteration()

        if self.finished:
            self.time_to_finish = self.time_elapsed

    def allocate_gpu(self, gpu_config : {}):
        '''input: a dictionary of Machine : GPU'''
        assert self.gpu_config == {}
        self.gpu_config.update(gpu_config)

    def deallocate_gpus(self):
        self.gpu_config.clear()

    def migrate_job(self, migrate_time : int):
        '''begins migration process for job; does not affect the underlying GPUs (migration called separately on those)'''
        self.time_until_free = migrate_time

    @property
    def progress(self):
        '''check progress as iter_completed/iter_total'''
        return self.iter_completed / self.iter_total

    @property
    def finished(self):
        '''check whether job is finished (equivalent to progress >= 1)'''
        return self.iter_completed >= self.iter_total

    @property
    def num_gpus(self):
        '''number of gpus (ignoring configuration of machines)'''
        return sum(len(x) for x in self.gpu_config.values())

    @property
    def finish_time(self):
        '''the time at which the job specifically completed'''
        return self.start_time + self.time_to_finish

    @property
    def waiting(self):
        '''job is completely idle (not running nor paused)'''
        return (self.num_gpus == 0) and (self.time_until_free <= 0)

    @property
    def read_iter_completed(self):
        '''number of iterations completed'''
        return self.iter_completed

    @property
    def read_memory_usage(self):
        '''memory usage of total job'''
        return self.memory_usage

    @property
    def read_gpu_config(self):
        '''gpu config in form of Machine : GPU'''
        return self.gpu_config