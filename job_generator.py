from job import Job
import scipy.stats as ss
import random
import pandas as pd
import numpy as np

MAX_DURATION = 20000

def transform_data(df : pd.DataFrame):
    # data in duration, # gpus, # nodes
    df = df.iloc[:,[3,4,5]]

    df.drop(labels = df[df.iloc[:,0] > MAX_DURATION].index, inplace=True)
    df.reset_index(inplace=True, drop=True)
    df.iloc[:,0].clip(upper=MAX_DURATION, inplace = True)
    #remove when # gpus or duration = 0
    df.clip(lower = 1, inplace=True)
    return df

class Synthetic_Job_Generator:
    def __init__(self):
        self.internal_id = 1
        self.time_until_next_job = 0
        self.reset_timer()
        self.df = transform_data(pd.read_csv('data/pd_data.csv', sep=','))

    def reset_timer(self):
        self.time_until_next_job = int(ss.expon.rvs(scale=180)) #mean of 3 minutes is roughly accurate

    def create_job(self, timestamp, get_params = False):

        #GPU asked, duration, and time between jobs actually somewhat independent; can naively model so

        row = np.array(self.df.sample())[0] #duration, gpus, nodes
        sample_duration, sample_gpu = int(row[0]/2), row[1]
        rvs = [max(rv, 0) for rv in ss.norm.rvs(loc=1, scale=.15, size=2)]

        true_time = sample_duration * rvs[0] #sample noise
        user_time = true_time * rvs[1] #user estimate noise

        #[.25, .25, .25, .25, 0]
        #[.2, .1, .2, .25, .25]

        if random.random() < 1/4:
            gpu_req = random.choices([1,2,4,8], weights = [.6,.15,.05,.3], k = 1)[0]
        else:
            gpu_req = sample_gpu

        job_params = {
            'id' : self.internal_id, 'start_time' : timestamp, 'num_iterations' : 1000000,
            'user_time_estimates' : {gpu_req : user_time}, 'req_gpu' : gpu_req,
            'true_time_rates' : {gpu_req : true_time}
        }

        self.internal_id += 1
        self.reset_timer()

        if get_params:
            return job_params

        return Job(**job_params)

    @property
    def job_arrived(self):
        if self.time_until_next_job <= 0:
            return True
        else:
            self.time_until_next_job -= 1