#Utilities for data analysis on Cluster object/cluster fields
import numpy as np
from cluster import Cluster
from job import Job, extrapolate_estimate

MINUTES_PER_DAY = (24 * 60)
MICROSECONDS_PER_MINUTE = (60 * 1000)

def get_all_gpu_utilization(cluster : Cluster):
    '''defined as the number of hours that each gpu is used
    should be used in compound with get_average_factors'''
    x = []
    for machine in cluster.machine_array:
        x += machine.gpu_util_history
    return x

def get_average_factors(completed_jobs : [Job]):
    '''gets average factor of every job (excluding 0s)'''
    x = []
    for job in completed_jobs:
        y = []
        for z in job.allocation_history:
            if z[1]:
                y.append(z[1])

        x.append(np.average(y))

    return x

def get_FTF(completed_jobs : [Job], jobs_per_hour = 3.7, duration_per_job = 2.5, cluster_gpus = 16):
    return [job.time_to_finish / extrapolate_estimate(cluster_gpus / jobs_per_hour / duration_per_job, job.true_times)
            for job in completed_jobs]

def get_JCT(completed_jobs : [Job]):
    return [job.time_to_finish for job in completed_jobs]


def get_total_time(completed_jobs : [Job]):
    return [list(job.true_times.values())[0] for job in completed_jobs]