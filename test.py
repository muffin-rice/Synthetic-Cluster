from job_generator import Synthetic_Job_Generator
from cluster import Cluster
import os
import random
import pickle

if __name__ == '__main__':
    random.seed(1000)
    SJG = Synthetic_Job_Generator()
    cluster = Cluster(num_gpus=50)
    current_time = 0

    #create batch of jobs

    cluster.queue_job(SJG.create_job(current_time))
    while not cluster.resting or current_time < 1000:
        if current_time < 1000 and random.random() < .15:
            cluster.queue_job(SJG.create_job(current_time))

        current_time += 1
        cluster.tick()

    pickle.dump(cluster, open('data/test-1.pkl', 'wb'))