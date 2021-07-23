from job_generator import Synthetic_Job_Generator, Synthetic_Job_Generator2
import pickle

NUM_JOBS_TO_GENERATE = 200

if __name__ == '__main__':
    SJG = Synthetic_Job_Generator2()
    t = 1
    jobs = [SJG.create_job(0, get_params=True)]

    for i in range(NUM_JOBS_TO_GENERATE-1):
        while not SJG.job_arrived:
            t+=1

        jobs.append(SJG.create_job(t, get_params=True))

    print(t)

    pickle.dump(jobs, open('data/job_simul_1.pkl', 'wb'))