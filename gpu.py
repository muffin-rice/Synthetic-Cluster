from copy import deepcopy

UPDATE_FREQ = 10

class GPU:
    def __init__(self, id : int, total_memory : int):
        self.id = id
        self.total_memory = total_memory
        self.usage = 0
        self.status = 0 #status = 1 when it is in use, 0 otherwise
        #job_history[i] is the job history at tick i (could be empty set)
        self.job_history = [] #[{job_id}]
        self.current_jobs = set()
        self.pause_time = 0
        self.update_i = UPDATE_FREQ

    def __hash__(self):
        return self.id.__hash__()

    def __str__(self):
        if self.free:
            return f'GPU {self.id}, FREE'

        return f'GPU {self.id}, RUN job {self.current_jobs}'

    def record_jobs(self):
        if self.update_i <= 0:
            self.update_i = UPDATE_FREQ
            self.job_history.append(deepcopy(self.current_jobs))
        else:
            self.update_i -= 1

    def tick(self):
        if self.pause_time > 0:
            self.pause_time -= 1
        self.record_jobs()

    def assign_job(self, job_id):
        #TODO: packing
        self.current_jobs.add(job_id)

    def remove_job(self, job_id):
        self.current_jobs.remove(job_id)

    def get_all_unique_jobs(self):
        #returns all distinct job ids that were done
        return set.union(*self.job_history)

    def get_gpu_utilization(self):
        #return percentage of time the gpu was not idle
        #TODO: packing/memory considerations
        try:
            return sum(job_ids != set() for job_ids in self.job_history) / len(self.job_history)
        except:
            return 0

    def pause(self, time): #pause the GPU for penalty/migration things
        self.pause_time = time

    @property
    def free(self):
        if self.pause_time > 0:
            return False
        return self.current_jobs == set()
