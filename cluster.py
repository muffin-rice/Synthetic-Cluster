from job import Job
from gpu import GPU

LEASE_TIME = 10

class Cluster:
    def __init__(self, num_gpus : int): #just a naive cluster with num_gpus identical GPUs
        self._gpu_array = [GPU(i, 100) for i in range(num_gpus)]
        self.free_gpus = [gpu for gpu in self.gpu_array]
        self.waiting_jobs, self.current_jobs, self.finished_jobs = set(), set(), []
        self.current_leases = {}
        self.wait_hist, self.curr_hist, self.finish_hist = [], [], []

    def queue_job(self, job : Job):
        self.waiting_jobs.add(job)

    def check_leases(self):
        for job in self.current_jobs:
            if self.current_leases[job] == 0:


    def remove_job(self, job):
        self.current_jobs.remove(job)
        self.finished_jobs.append(job)
        self.current_leases.pop(job)

        self.free_gpus = []
        for gpu in self.gpu_array:
            if gpu.free:
                self.free_gpus.append(gpu)

        self.free_gpus += list(job.allocated_gpus)

    def tick(self):
        # TODO: implement "lease"
        assert self.current_jobs.intersection(self.waiting_jobs) == set()

        #schedule algorithm
        #self.schedule_jobs_max()
        self.schedule_jobs_time()
        self.check_leases()

        for obj in (self.gpu_array + list(self.waiting_jobs) + list(self.current_jobs)):
            obj.tick()

        for job in list(self.current_jobs):
            if job.finished: #gpu status already changed
                self.remove_job(job)

        self.wait_hist.append(len(self.waiting_jobs))
        self.curr_hist.append(len(self.current_jobs))
        self.finish_hist.append(len(self.finished_jobs))

    #THE SCHEDULER SHOULD:
    #assign waiting jobs to the free gpus;
    #remove assigned jobs from waiting_jobs and assigned gpus from free_gpus

    def schedule_jobs_max(self):

        #implement simple max->min algo; just schedule the jobs that need the most gpus first
        sorted_jobs = sorted(self.waiting_jobs, key = lambda job : job.req_gpu, reverse = True)

        for job in sorted_jobs:
            if job.req_gpu < len(self.free_gpus):
                assigned_gpus = self.free_gpus[:job.req_gpu]
                for gpu in assigned_gpus:
                    gpu.assign_job(job.id)
                    #print(gpu.current_jobs)

                job.allocate_gpus(set(assigned_gpus))
                self.current_leases[job] = LEASE_TIME

                self.waiting_jobs.remove(job)
                self.current_jobs.add(job)
                self.free_gpus = self.free_gpus[job.req_gpu:]

    def schedule_jobs_time(self):
        sorted_jobs = sorted(self.waiting_jobs, key = lambda job : job.est_finish_time - job.start_time, reverse=True)

        for job in sorted_jobs:
            if job.req_gpu < len(self.free_gpus):
                assigned_gpus = self.free_gpus[:job.req_gpu]
                for gpu in assigned_gpus:
                    gpu.assign_job(job.id)
                    #print(gpu.current_jobs)

                job.allocate_gpus(set(assigned_gpus))

                self.waiting_jobs.remove(job)
                self.current_jobs.add(job)
                self.free_gpus = self.free_gpus[job.req_gpu:]

    @property
    def resting(self):
        return (self.waiting_jobs == set()) and (self.current_jobs == set())

    @property
    def gpu_array(self):
        return self._gpu_array