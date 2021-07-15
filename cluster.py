from job import Job
from gpu import GPU

LEASE_TIME = 600
LEASE_PENALTY = 0
UPDATE_FREQ = 10

class Cluster:
    def __init__(self, num_gpus : int): #just a naive cluster with num_gpus identical GPUs
        self._gpu_array = [GPU(i, 100) for i in range(num_gpus)]
        self.free_gpus = [gpu for gpu in self.gpu_array]
        self.waiting_jobs, self.current_jobs, self.finished_jobs = set(), set(), []
        self.current_leases = {} #job : lease_time
        self.wait_hist, self.curr_hist, self.finish_hist, self.update_i = [], [], [], UPDATE_FREQ

    def queue_job(self, job : Job):
        self.waiting_jobs.add(job)

    def deallocate_gpus_from_job(self, job : Job):
        #removes job from current job and lease; used when job finishes and when lease over
        #removes job ids from gpus (thereby updating gpu status) and updates self.free_gpu list
        self.current_leases.pop(job)
        self.current_jobs.remove(job)

        for gpu in job.allocated_gpus:
            gpu.remove_job(job.id)
            if not job.finished:
                gpu.pause(LEASE_PENALTY)

        job.deallocate_gpus()

        self.free_gpus = []
        for gpu in self.gpu_array:
            if gpu.free:
                self.free_gpus.append(gpu)

    def allocate_gpus_to_job(self, job, gpus):
        #adds a job
        for gpu in gpus:
            gpu.assign_job(job.id)

        job.allocate_gpus(set(gpus))
        self.current_leases[job] = LEASE_TIME

        self.waiting_jobs.remove(job)
        self.current_jobs.add(job)
        self.free_gpus = self.free_gpus[job.req_gpu:]

    def check_leases(self):
        #checks the leases; removes from current if lease is over, adds to waiting
        for job in {x for x in self.current_jobs}:
            if self.current_leases[job] <= 0: #if lease is over
                self.deallocate_gpus_from_job(job)
                self.waiting_jobs.add(job)
            else: #decrement lease
                self.current_leases[job] -= 1

    def complete_job(self, job):
        self.deallocate_gpus_from_job(job)
        self.finished_jobs.append(job)

    def record_jobs(self):
        if self.update_i <= 0:
            self.wait_hist.append(len(self.waiting_jobs))
            self.curr_hist.append(len(self.current_jobs))
            self.finish_hist.append(len(self.finished_jobs))
            self.update_i = UPDATE_FREQ

        else:
            self.update_i -= 1

    def tick(self):
        assert not self.current_jobs.intersection(self.waiting_jobs)

        #schedule algorithm
        #self.schedule_jobs_max()
        #self.schedule_jobs_time()
        self.schedule_jobs_fairness_ratio()
        self.check_leases()

        for obj in (self.gpu_array + list(self.waiting_jobs) + list(self.current_jobs)):
            obj.tick()

        for job in list(self.current_jobs):
            if job.finished:
                self.complete_job(job)

        self.record_jobs()


    #THE SCHEDULER SHOULD:
    #assign waiting jobs to the free gpus;
    #remove assigned jobs from waiting_jobs and assigned gpus from free_gpus

    def schedule_jobs_max(self):

        #implement simple max->min algo; just schedule the jobs that need the most gpus first
        sorted_jobs = sorted(self.waiting_jobs, key = lambda job : job.req_gpu, reverse = True)

        for job in sorted_jobs:
            if job.req_gpu < len(self.free_gpus):
                assigned_gpus = self.free_gpus[:job.req_gpu]
                self.allocate_gpus_to_job(job, assigned_gpus)

    def schedule_jobs_time(self):
        sorted_jobs = sorted(self.waiting_jobs, key = lambda job : job.est_finish_time - job.start_time, reverse=True)

        for job in sorted_jobs:
            if job.req_gpu < len(self.free_gpus):
                assigned_gpus = self.free_gpus[:job.req_gpu]
                self.allocate_gpus_to_job(job, assigned_gpus)

    def schedule_jobs_fairness_ratio(self):
        sorted_jobs = sorted(self.waiting_jobs, key = lambda job : job.fairness_ratio, reverse=True)

        for job in sorted_jobs:
            if job.req_gpu < len(self.free_gpus):
                assigned_gpus = self.free_gpus[:job.req_gpu]
                self.allocate_gpus_to_job(job, assigned_gpus)

    @property
    def resting(self):
        return (self.waiting_jobs == set()) and (self.current_jobs == set())

    @property
    def gpu_array(self):
        return self._gpu_array