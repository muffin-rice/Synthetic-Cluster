from copy import deepcopy

class GPU:
    def __init__(self, id : int, total_memory : int):
        self.id = id
        self.total_memory = total_memory
        self.usage = 0
        self.status = 0 #status = 1 when it is in use, 0 otherwise
        #job_history[i] is the job history at tick i (could be empty set)
        self.job_history = [] #[{job_id}]
        self.current_jobs = set()

    def __hash__(self):
        return self.id.__hash__()

    def __str__(self):
        if self.free:
            return f'GPU {self.id}, FREE'

        return f'GPU {self.id}, RUN job {self.current_jobs}'

    def tick(self):
        self.job_history.append(deepcopy(self.current_jobs))

    def assign_job(self, job_id):
        #TODO: packing
        self.current_jobs.add(job_id)
        self.status = 1

    def remove_job(self, job_id):
        self.current_jobs.remove(job_id)
        self.status = self.current_jobs != set() #0 if empty set, 1 otherwise

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

    @property
    def free(self):
        return self.status