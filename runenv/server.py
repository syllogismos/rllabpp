from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import argparse
import urllib.parse as urlparse

# from osim.env import RunEnv
import numpy as np
from runenv.helpers import Scaler
import multiprocessing
import pickle, os, joblib
import random
import tensorflow as tf

from rllab.sampler.utils import rollout
from rllab.envs.gym_env import GymEnv
from rllab.envs.normalized_env import normalize
from sandbox.rocky.tf.envs.base import TfEnv

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


PORT_NUMBER = 8018

def dump_episodes(env_name, difficulty,
        chk_dir, batch_size, cores):
    # scaler_file = chk_dir + '/scaler_latest'
    # scaler = pickle.load(open(scaler_file, 'rb'))
    p = multiprocessing.Pool(cores, maxtasksperchild=1)
    paths = p.map(get_paths_from_latest_policy,
            [(env_name, difficulty, 
              chk_dir, batch_size//cores)]*cores)
    p.close()
    p.join()
    paths = sum(paths, [])
    episodes_file = chk_dir + '/episodes_latest'
    pickle.dump(paths, open(episodes_file, 'wb'))

def get_paths_from_latest_policy(pickled_obj):
    """
    pickled_obj = (env_name, chk_dir, batch_size_per_core)
    """
    env_name = pickled_obj[0]
    difficulty = pickled_obj[1]
    chk_dir = pickled_obj[2]
    batch_size_per_core = pickled_obj[3]
    with tf.Session() as sess:
        data = joblib.load(chk_dir + 'params.pkl')
        policy = data['policy']
        env = TfEnv(GymEnv(env_name, difficulty=difficulty,
            runenv_seed=1, visualize=False,
            record_log=False, record_video=False))
        total_length = 0
        paths = []
        while total_length < batch_size_per_core:
            path = rollout(env, policy, max_path_length=1000,
                animated=False, always_return_paths=True)
            paths.append(path)
            total_length += len(path['rewards'])
    return paths

class myHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if '/ping' in self.path:
            print(self.path)
            parsed_url = urlparse.urlparse(self.path)
            print(urlparse.parse_qs(parsed_url.query))
            print('lmao it worked')
            self.send_response(200)
            self.send_header('Content-type', 'application/javascript')
            self.end_headers()
            self.wfile.write(bytes(json.dumps({'anil': 'tanu'}), 'utf8'))
            return
        elif '/get_paths' in self.path:
            print(self.path)
            parsed_url = urlparse.urlparse(self.path)
            query = urlparse.parse_qs(parsed_url.query)
            env_name = query['env_name'][0]
            chk_dir = query['chk_dir'][0]
            batch_size = int(query['batch_size'][0])
            cores = int(query['cores'][0])
            difficulty = int(query['difficulty'][0])
            dump_episodes(env_name, difficulty, chk_dir, batch_size, cores)
            self.send_response(200)
            self.send_header('Content-type', 'application/javascript')
            self.end_headers()
            self.wfile.write(bytes(json.dumps({'Success': 'OK'}), 'utf8'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--listen', type=str, default='127.0.0.1')
    parser.add_argument('--port', type=int, default=PORT_NUMBER)
    args = parser.parse_args()
    server = HTTPServer((args.listen, args.port), myHandler)
    print('Server started on', args)
    server.serve_forever()