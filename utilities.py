import datetime
import numpy as np
from cluster import Cluster

DATE_FORMAT_STR = '%Y-%m-%d %H:%M:%S'
MINUTES_PER_DAY = (24 * 60)
MICROSECONDS_PER_MINUTE = (60 * 1000)

def parse_date(date_str):
    """Parses a date string and returns a datetime object if possible.

       Args:
           date_str: A string representing a date.

       Returns:
           A datetime object if the input string could be successfully
           parsed, None otherwise.
    """
    if date_str is None or date_str == '' or date_str == 'None':
        return None
    return datetime.datetime.strptime(date_str, DATE_FORMAT_STR)


def timedelta_to_minutes(timedelta):
    """Converts a datetime timedelta object to minutes.

       Args:
           timedelta: The timedelta to convert.

       Returns:
           The number of minutes captured in the timedelta.
    """
    minutes = 0.0
    minutes += timedelta.days * MINUTES_PER_DAY
    minutes += timedelta.seconds / 60.0
    minutes += timedelta.microseconds / MICROSECONDS_PER_MINUTE
    return minutes


def round_to_nearest_minute(t):
    """Rounds a datetime object down to the nearest minute.

       Args:
           t: A datetime object.

        Returns:
            A new rounded down datetime object.
    """
    return t - datetime.timedelta(seconds=t.second, microseconds=t.microsecond)


def add_minute(t):
    """Adds a single minute to a datetime object.

       Args:
           t: A datetime object.

        Returns:
            A new datetime object with an additional minute.
    """
    return t + datetime.timedelta(seconds=60)


def get_cdf(data):
    """Returns the CDF of the given data.

       Args:
           data: A list of numerical values.

       Returns:
           An pair of lists (x, y) for plotting the CDF.
    """
    sorted_data = sorted(data)
    p = 100. * np.arange(len(sorted_data)) / (len(sorted_data) - 1)
    return sorted_data, p


def get_bucket_from_num_gpus(num_gpus):
    """Maps GPU count to a bucket for plotting purposes."""
    if num_gpus is None:
        return None
    if 0 < num_gpus < 2:
        return 0
    elif 2 <= num_gpus <= 4:
        return 1
    elif 5 <= num_gpus <= 8:
        return 2
    elif num_gpus > 8:
        return 3
    else:
        return None


def get_plot_config_from_bucket(bucket):
    """Returns plotting configuration information."""
    if bucket == 0:
        return ('1', 'green', '-')
    elif bucket == 1:
        return ('2-4', 'blue', '-.')
    elif bucket == 2:
        return ('5-8', 'red', '--')
    elif bucket == 3:
        return ('>8', 'purple', ':')

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