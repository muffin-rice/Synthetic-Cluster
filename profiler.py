from machine import Machine
from job import Job

class Profiled_Job:
    '''mirrors the job'''
    def __init__(self, user_estimates : {}, job):
        #stats in the form of gpu_config : [total time, iters completed/progress, average memory usage (?) across gpus]
        self._stats = {}
        self.job = job
        self.current_iteration = [0, 0, 0, 0, 0] #[gpu_config, total_time, iters_started, iters_now, mem]
        self.user_estimates = user_estimates
        self.current_marker = 0
        self.estimated_values = []

    def start_profiling(self):
        '''remake current_iteration, restart tracking'''
        self.current_marker = 1
        hash_gpu_config = (len(self.job.gpu_config.keys()), sum(len(x) for x in self.job.gpu_config.values()))
        self.current_iteration = [hash_gpu_config, 0, self.job.read_iter_completed, self.job.read_iter_completed, 0]

    def stop_profiling(self):
        '''record current iteration into stats'''
        self._stats = self.stats
        #remove current iteration
        self.current_iteration.clear()
        self.current_marker = 0

    def tick(self):
        self.current_iteration[1] += 1
        self.current_iteration[3] = self.job.read_iter_completed
        self.current_iteration[4] += self.job.read_memory_usage

    @property
    def stats(self):
        stats_copy = {k: v for k, v in self._stats.items()}
        k, v = self.current_iteration[0], self.current_iteration[1:]

        if k in stats_copy:
            stats_copy[k][0] += v[0]
            stats_copy[k][1] += v[2] - v[1]
            stats_copy[k][2] += v[3]

        else:
            stats_copy[k] = [v[0], v[2] - v[1], v[3]]

        return stats_copy


class Profiler:
    '''goal is that given the gpu usage and the job's continuous progress,
    guesstimate the job's model size and the job type (perhaps length as well?)'''

    '''considerations: 
    for packed jobs, there may be interference 
    noise in behavior 
    '''
    def __init__(self, machine_array : [Machine]):
        '''keeps track of machine, gpu, job stats; mostly mirrors what is inside cluster'''
        self._machine_array = machine_array

        #job : data
        self.stats = {}

        #build a graph for compatible jobs?

    def add_job(self, job : Job, user_estimates : {}):
        self.stats[job] = Profiled_Job(user_estimates, job)

    def _update_current_markers(self, current_jobs : set):
        '''changes self.stats for recording purposes'''
        old_jobs = set()
        finished_jobs = set()

        for job, pj in self.stats.items():
            if pj.current_marker == 1:
                if job.finished: #remove if any jobs finished
                    finished_jobs.add(job)
                else: #find all currently running jobs
                    old_jobs.add(job)

        for job in finished_jobs:
            self.stats.pop(job) #remove all finished jobs

        for job in old_jobs.difference(current_jobs): #jobs that were paused
            self.stats[job].stop_profiling()

        for job in current_jobs.difference(old_jobs): #jobs that were newly resumed
            self.stats[job].start_profiling()

    def _update_stats(self, current_jobs : set):
        '''updates self.stats'''
        for job in current_jobs:
            self.stats[job].tick()

    def tick(self, current_jobs : set):
        '''profiler should tick before job ticks but after scheduler ticks '''
        self._update_current_markers(current_jobs)
        self._update_stats(current_jobs)