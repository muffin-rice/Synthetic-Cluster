from job_generator import Synthetic_Job_Generator
from cluster import Cluster
import os
import random
import pickle

PERIOD_OF_JOBS = 20000

if __name__ == '__main__':
    random.seed(1000)
    SJG = Synthetic_Job_Generator()
    cluster = Cluster(num_gpus=24)
    current_time = 0

    cluster.queue_job(SJG.create_job(current_time))
    while not cluster.resting or current_time < PERIOD_OF_JOBS:
        if not current_time%10000:
            print(f'Waiting: {len(cluster.waiting_jobs)}')

        if current_time < PERIOD_OF_JOBS and SJG.job_arrived:
            cluster.queue_job(SJG.create_job(current_time))

        current_time += 1
        cluster.tick()

    print(current_time)

    pickle.dump(cluster, open('data/test-1.pkl', 'wb'))