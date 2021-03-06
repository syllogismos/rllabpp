import numpy as np
import subprocess, psutil, time


class Scaler(object):
    """ Generate scale and offset based on running mean and stddev along axis=0

        offset = running mean
        scale = 1 / (stddev + 0.1) / 3 (i.e. 3x stddev = +/- 1.0)
    """

    def __init__(self, obs_dim):
        """
        Args:
            obs_dim: dimension of axis=1
        """
        self.vars = np.zeros(obs_dim)
        self.means = np.zeros(obs_dim)
        self.m = 0
        self.n = 0
        self.first_pass = True

    def update(self, x):
        """ Update running mean and variance (this is an exact method)
        Args:
            x: NumPy array, shape = (N, obs_dim)

        see: https://stats.stackexchange.com/questions/43159/how-to-calculate-pooled-
               variance-of-two-groups-given-known-group-variances-mean
        """
        if self.first_pass:
            self.means = np.mean(x, axis=0)
            self.vars = np.var(x, axis=0)
            self.m = x.shape[0]
            self.first_pass = False
        else:
            n = x.shape[0]
            new_data_var = np.var(x, axis=0)
            new_data_mean = np.mean(x, axis=0)
            new_data_mean_sq = np.square(new_data_mean)
            new_means = ((self.means * self.m) +
                         (new_data_mean * n)) / (self.m + n)
            self.vars = (((self.m * (self.vars + np.square(self.means))) +
                          (n * (new_data_var + new_data_mean_sq))) / (self.m + n) -
                         np.square(new_means))
            # occasionally goes negative, clip
            self.vars = np.maximum(0.0, self.vars)
            self.means = new_means
            self.m += n

    def get(self):
        """ returns 2-tuple: (scale, offset) """
        return 1 / (np.sqrt(self.vars) + 0.1) / 3, self.means


def start_env_server(port, p=0, ec2=True):
    port = str(int(port) + p)
    http_server_path = '/home/ubuntu/rllabpp/runenv/server.py'
    command = http_server_path + ' --port ' + port
    process = subprocess.Popen(command, shell=True)
    time.sleep(5)
    return process.pid


def destroy_env_server(pid):
    try:
        process = psutil.Process(pid)
        for child in process.children():
            child.terminate()
            child.wait()
            process.terminate()
            process.wait()
    except:
        print("process doesnt exist", pid)
        pass
    time.sleep(5)
