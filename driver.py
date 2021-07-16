from cluster import Cluster
import random
import pickle
from job import Job

PERIOD_OF_JOBS = 20000 #period of timeat which jobs are continuously incoming

if __name__ == '__main__':
    random.seed(1000)
    jobs = pickle.load(open('data/test-1.pkl', 'rb'))
    cluster = Cluster(num_gpus=24)
    current_time = 0

    cluster.queue_job(Job(**jobs[0]))
    curr_job_index = 1

    while not cluster.resting or current_time < PERIOD_OF_JOBS:
        if not current_time%10000:
            print(f'Waiting: {len(cluster.waiting_jobs)}')

        if current_time == jobs[curr_job_index].start_time:
            cluster.queue_job(Job(**jobs[curr_job_index]))
            curr_job_index += 1

        current_time += 1
        cluster.tick()

    print(current_time)

    pickle.dump(cluster, open('data/test-1.pkl', 'wb'))