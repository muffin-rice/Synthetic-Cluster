from job import Job
import scipy.stats as ss
import random

class Synthetic_Job_Generator:
    def __init__(self):
        self.internal_id = 1
        pass

    def create_job(self, timestamp):

        times = ss.norm.rvs(loc = 250, scale = 100, size=2)
        user_time, true_time = max(times[0], 10), max(times[1], 10)

        #[.25, .25, .25, .25, 0]
        #[.2, .1, .2, .25, .25]
        gpu_req = random.choices([1,2,4,8,12], weights = [.15, .1, .1, .4, .25], k = 1)[0]

        new_job = Job(id=self.internal_id, start_time=timestamp, num_iterations=1000000,
                      user_time_estimates={gpu_req : user_time}, req_gpu=gpu_req,
                      true_time_rates={gpu_req : true_time})

        self.internal_id += 1

        return new_job