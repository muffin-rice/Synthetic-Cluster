from job import Job
from machine import Machine
from profiler import Profiler
import copy

LEASE_TIME = 600000
LEASE_PENALTY = 0
UPDATE_FREQ = 10

def schedule_jobs_FIFO(free_resources : {int : [int]}, idle_jobs : {Job}, profiler : Profiler) -> {Job : {int : [int]}}: 
    '''simple FIFO; by job.start_time. baseline scheduler - no lease'''
    #free resources is machine_id : gpu_id 
    #returns a gpu_config (in ids) for each job (only returns it for some jobs)
    sorted_jobs = sorted(idle_jobs, key = lambda job : job.start_time)

    # simply assign by max preferred gpus 
    ans = {}
    while True: #iterate through jobs 
        if not sorted_jobs: 
            break 

        curr_job = sorted_jobs.pop(0)
        gpu_left = sum(len(x) for x in free_resources.values())

        if curr_job.pref_gpus[0] > gpu_left: #cannot allocate enough for job
            break 

        # temp dict is {mach_id : [gpu_ids]}
        temp_dict = {}

        # number of gpus to allocate
        gpus_to_allocate = min(curr_job.pref_gpus[1], gpu_left)

        # build dictionary based on the number of gpus to allocate
        free_resources2 = {k1 : v1  for k1, v1 in free_resources.items()}
        for k, v in free_resources2.items(): # machine : gpu ids
            if gpus_to_allocate == 0: 
                break
            if len(v) <= gpus_to_allocate: 
                temp_dict[k] = copy.deepcopy(v)
                gpus_to_allocate -= len(v)
                free_resources.pop(k)
            else: # partial allocation on the machine 
                temp_dict[k] = copy.deepcopy(v[:gpus_to_allocate])
                free_resources[k] = v[gpus_to_allocate:]
                v = v[gpus_to_allocate:]
                break
        
        ans[curr_job] = temp_dict

    return ans



def assert_no_overlap(*job_sets):
    for i in range(len(job_sets)-1):
        for j in range(i+1, len(job_sets)):
            x = job_sets[i].intersection(job_sets[j])
            if x:
                print(i,j)
                print(x)
                raise AssertionError('Sets have intersection')

class Cluster:
    def __init__(self, num_machines : int): 
        self._machine_array = [Machine(i) for i in range(num_machines)]
        self.waiting_jobs, self.current_jobs, self.migrating_jobs, self.finished_jobs = set(), set(), set(), []
        self.current_leases = {} #job : lease_time
        self.wait_hist, self.curr_hist, self.finish_hist, self.update_i = [], [], [], UPDATE_FREQ
        self.profiler = Profiler(self._machine_array)

    def queue_job(self, job : Job, user_estimates : {}):
        self.waiting_jobs.add(job)
        self.profiler.add_job(job, user_estimates)

    def deallocate_gpus_from_job(self, job : Job):
        '''removes job from current job and lease; called when either job finishes or when lease over'''
        #when removing a job, the machines/gpus are removed from Job object and Job objected removed from
        #Machine object
        self.current_leases.pop(job)
        self.current_jobs.remove(job)

        for mach in job.gpu_config.keys():

            if not job.finished: #in migration
                mach.migrate_job(job, LEASE_PENALTY)
                self.migrating_jobs.add(job)
                job.migrate_job(LEASE_PENALTY)

            mach.remove_job(job)

        job.deallocate_gpus()

    def allocate_gpus_to_job(self, job : Job, gpu_config : {Machine : [int]}):
        '''assigns a particular gpu configuration to job'''
        for mach,v in gpu_config.items():
            mach.assign_job(job, v)

        job.allocate_gpu(gpu_config)
        self.current_leases[job] = LEASE_TIME
        self.waiting_jobs.remove(job)
        self.current_jobs.add(job)

    def check_leases(self):
        '''checks the leases; removes from current if lease is over, adds to waiting'''
        for job in {x for x in self.current_jobs}:
            if self.current_leases[job] <= 0: #if lease is over
                self.deallocate_gpus_from_job(job)
            else: #decrement lease
                self.current_leases[job] -= 1

        for job in {x for x in self.migrating_jobs}:
            if job.waiting:
                self.waiting_jobs.add(job)
                self.migrating_jobs.remove(job)

    def complete_job(self, job : Job):
        '''cleans up a job (add to finished_jobs) after total completion'''
        self.deallocate_gpus_from_job(job)
        self.finished_jobs.append(job)

    def record_jobs(self):
        if self.update_i <= 0:
            self.wait_hist.append(len(self.waiting_jobs))
            self.curr_hist.append(len(self.current_jobs) + len(self.migrating_jobs))
            self.finish_hist.append(len(self.finished_jobs))
            self.update_i = UPDATE_FREQ

        else:
            self.update_i -= 1

    def _parse_scheduler_return_format(self, jobs_to_sched : {Job : {int : [int]}}):
        '''parses out scheduler return format and assigns all the jobs'''
        for job, gpu_config in jobs_to_sched.items():
            self.allocate_gpus_to_job(job, {self.machine_array[mach_id] : gpu_ids for mach_id, gpu_ids in gpu_config.items()})

    def tick(self):
        assert_no_overlap(self.waiting_jobs, self.current_jobs, set(self.finished_jobs), self.migrating_jobs)

        #schedule algorithm
        jobs_to_sched = schedule_jobs_FIFO(self.free_resources, self.waiting_jobs, self.profiler)

        self._parse_scheduler_return_format(jobs_to_sched)

        self.check_leases()
        self.profiler.tick(self.current_jobs)

        for obj in (self._machine_array + list(self.waiting_jobs) + list(self.current_jobs) + list(self.migrating_jobs)):
            obj.tick()

        for job in list(self.current_jobs):
            if job.finished:
                self.complete_job(job)

        self.record_jobs()

    @property
    def free_resources(self): 
        fresources = {}
        for mach in self._machine_array: 
            if mach.free_whole_gpus: 
                fresources[mach.id] = list(gpu.id for gpu in mach.free_whole_gpus)
        
        return fresources

    @property
    def resting(self):
        return (self.waiting_jobs == set()) and (self.current_jobs == set())

    @property
    def machine_array(self):
        return self._machine_array