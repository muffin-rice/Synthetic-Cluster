from cluster import Cluster
import random
import pickle
from job import Job

PERIOD_OF_JOBS = 20000 #period of timeat which jobs are continuously incoming

if __name__ == '__main__':
    random.seed(1000)
    jobs = pickle.load(open('data/job_simul_1.pkl', 'rb'))
    cluster = Cluster(num_machines=5)
    current_time = 0

    x = jobs[0]
    user_est = x['user_estimates']
    x.pop('user_estimates')
    cluster.queue_job(Job(**x), user_estimates=user_est)

    curr_job_index = 1
    print('Time\t# Paused\t# Finished')

    while not cluster.resting or curr_job_index < len(jobs):
        if not current_time%10000:
            print(f'{current_time}\t{len(cluster.waiting_jobs)}\t\t{len(cluster.finished_jobs)}')

        while curr_job_index < len(jobs) and current_time == jobs[curr_job_index]['start_time']:
            curr_job = jobs[curr_job_index]
            user_est = curr_job['user_estimates']
            curr_job.pop('user_estimates')
            cluster.queue_job(Job(**curr_job), user_estimates=user_est)
            curr_job_index += 1

        current_time += 1
        cluster.tick()

    print(current_time)

    pickle.dump(cluster, open('data/test-3.pkl', 'wb'))
    #test 1: FIFO lease
    #test 2: FIFO no penalty
    #test 3: FIFO no lease