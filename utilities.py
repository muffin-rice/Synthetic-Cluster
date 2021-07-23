#Utilities for data analysis on cluster 
import numpy as np
from cluster import Cluster

MINUTES_PER_DAY = (24 * 60)
MICROSECONDS_PER_MINUTE = (60 * 1000)

def get_all_gpu_utilization(cluster : Cluster):
    return [gpu.get_gpu_utilization() for gpu in cluster.gpu_array]

def get_job_finish_stats(cluster : Cluster):
    '''

    :param cluster:
    :return: [(start_time, projected_finish_time, true_finish_time)]
    '''
    return [(job.start_time, job.est_finish_time, job.finish_time) for job in cluster.finished_jobs]

def get_utilization_by_time(cluster : Cluster):
    #cluster
    total_time = len(cluster.gpu_array[-1].job_history)
    total_gpus = len(cluster.gpu_array)
    #avg_gpu_utilization = []
    #for i in range(total_time):
#
    #    x = sum([1 if gpu.job_history[i] else 0 for gpu in cluster.gpu_array])
    #    avg_gpu_utilization.append(x/total_gpus)

    avg_gpu_utilization = [sum([1 if gpu.job_history[i] else 0 for gpu in cluster.gpu_array])/total_gpus for i in range(total_time)]

    return avg_gpu_utilization

def get_fairness_ratios(cluster : Cluster):
    return [(stat[2]-stat[0]) / (stat[1]-stat[0]) for stat in get_job_finish_stats(cluster)]

def final_time(cluster : Cluster):
    return