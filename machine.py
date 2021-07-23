UPDATE_FREQ = 10

class GPU:
    def __init__(self, id : int, total_memory : int = 100):
        self.id, self.total_memory = id, total_memory #100 baseline as a percentage

        #job_history[i] is the job history [{job_id}] at tick i*update_freq (could be empty set)
        # percent_util[i] is the percent util [util] at tick i * update_freq (could be 0)
        self.job_history, self.percent_util = [], []
        
        #current_jobs is {Job}
        self.current_jobs = set()

        self.pause_time = 0
        self.update_i = UPDATE_FREQ

    def __hash__(self):
        x = f'GPU{self.id}'
        return x.__hash__()

    def __str__(self):
        if self.free:
            return f'GPU {self.id}, FREE'

        return f'GPU {self.id}, RUN job {self.current_jobs}'

    def __contains__(self, job):
        return job in self.current_jobs

    def record_jobs(self):
        '''records jobs in [[current job_id #1, ..., current job_id #n-1, free_memory]] format'''
        if self.update_i <= 0:
            curr_job_stats = [job.id for job in self.current_jobs]
            curr_job_stats.append(self.gpu_utilization)
            self.update_i = UPDATE_FREQ
            self.job_history.append(curr_job_stats)
        else:
            self.update_i -= 1

    def tick(self):
        '''tick method, used for recordkeeping + migration tracking'''
        if self.pause_time > 0:
            self.pause_time -= 1
        self.record_jobs()

    def assign_job(self, job):
        '''simply adds job to current job set'''
        self.current_jobs.add(job)

    def remove_job(self, job):
        '''removes specified job from current job set'''
        self.current_jobs.remove(job)

    def pause(self, time): #pause the GPU for penalty/migration
        self.pause_time = time

    def get_gpu_interference(self):
        '''interference for two 1-GPU tasks (packing only) - currently returns 1 '''
        if self.num_jobs < 2:
            return 1

        return 1

    def start_migration(self, migration_cost):
        #starts the migration for all the jobs in GPU
        #the migration may be different depending on the sze of the jobs/models
        self.pause_time = migration_cost

    @property
    def free(self):
        '''no current jobs, not in migration'''
        if self.pause_time > 0:
            return False
        return self.current_jobs == set()

    @property
    def all_unique_jobs(self):
        '''returns all distinct job ids that were done over history'''
        return set.union(*self.job_history)

    @property
    def gpu_utilization(self):
        '''percentage of memory used for packing'''
        return sum(job.memory_usage for job in self.current_jobs) / self.total_memory

    @property
    def total_gpu_usage_history(self):
        '''usage=1 when there is a single job (ignoring packing), percentage over total history'''
        try:
            return sum(len(x) != 1 for x in self.job_history) / len(self.job_history)
        except:
            return 0

    @property
    def available_for_packing(self):
        '''unavailable for packing when one of the jobs is using multiple gpus'''
        return all(job.num_gpus == 1 for job in self.current_jobs)

    @property
    def num_jobs(self):
        '''current number of jobs'''
        return len(self.current_jobs)

class Machine:
    def __init__(self, id : int, num_gpus : int = 4):
        self.id = id

        #gpus stored as {GPU_id : GPU} (like lookup table)
        self.gpus = {10 * id + i : GPU(id = 10 * id + i, total_memory=100) for i in range(num_gpus)}
        #store which GPUs job uses (on this machine) with Job : gpu_ids
        self.curr_jobs = {}

    def __hash__(self):
        x = f'Machine{self.id}'
        return x.__hash__()

    def assign_job(self, job, gpu_ids):
        '''assigns a job exclusively to GPU'''
        self.curr_jobs[job] = set()
        for gpu_id in gpu_ids: 
            curr_gpu = self.gpus[gpu_id]
            assert curr_gpu.free 
            curr_gpu.assign_job(job)
            self.curr_jobs[job].add(gpu_id)

    def remove_job(self, job):
        '''removes job from GPUs and current machine assignment'''
        assert job in self.curr_jobs.keys()
        for gpu_id in self.curr_jobs[job]:
            curr_gpu = self.gpus[gpu_id]
            assert job in curr_gpu #job should be in there 
            curr_gpu.remove_job(job)

        self.curr_jobs.pop(job)

    def migrate_job(self, job, migration_penalty):
        '''pauses all gpus in machine that contain the job to be migrated'''
        for gpu_id in self.curr_jobs[job]:
            curr_gpu = self.gpus[gpu_id]
            assert job in curr_gpu
            curr_gpu.pause(migration_penalty)

    def tick(self):
        for gpu in self.gpus.values():
            gpu.tick()

    def get_machine_locality_factor(self, job):
        '''has something to do with sockets and PCIe Switches (unknown - set to 1 for now)'''
        gpus_in_job = job.read_gpu_config[self]
        if len(gpus_in_job) == 1:
            return 1
        return 1

    def contain_gpu(self, gpu, int_lookup = False):
        '''checks whether certain gpu object is in (int_lookup if searching gpu_id)'''
        if int_lookup: 
            return gpu in self.gpus.keys()

        return gpu in self.gpus.values()

    @property
    def free_whole_gpus(self):
        '''set of GPUs (objects) that are completely free'''
        f = set()
        for gpu in self.gpus.values():
            if gpu.free:
                f.add(gpu)
        return f

    @property
    def indiv_gpu_utilization(self):
        '''returns a list of [GPU, util] - for packing specifially'''
        arr = [] #[gpu_id, amount_utilization]
        for gpu in self.gpus:
            if gpu.available_for_packing:
                arr.append([gpu.id, gpu.gpu_utilization])

        return arr
