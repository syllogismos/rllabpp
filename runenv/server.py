#! /home/ubuntu/anaconda2/envs/rllabpp/bin/python
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import argparse
import urllib.parse as urlparse

# from osim.env import RunEnv
import numpy as np
from runenv.helpers import Scaler
import multiprocessing
import pickle, os, joblib
import random, redis
import tensorflow as tf

from rllab.sampler.utils import rollout
from rllab.envs.gym_env import GymEnv
from rllab.envs.normalized_env import normalize
from sandbox.rocky.tf.envs.base import TfEnv

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


PORT_NUMBER = 8018

def dump_episodes(env_name, difficulty,
        chk_dir, batch_size, cores,
        max_obstacles, filter_type, history_len):
    scaler_file = os.path.join(chk_dir, 'scaler_latest')
    if os.path.exists(scaler_file): 
        scaler = pickle.load(open(scaler_file, 'rb'))
    else:
        scaler = None
    redis_conn = redis.Redis()
    redis_key = 'curr_batch_size-' + chk_dir
    redis_conn.set(redis_key, 0)
    p = multiprocessing.Pool(cores, maxtasksperchild=1)
    paths = p.map(get_paths_from_latest_policy,
            [(env_name, difficulty, max_obstacles, filter_type, history_len,
              chk_dir, scaler, batch_size//cores, batch_size)]*cores)
    p.close()
    p.join()
    paths = sum(paths, [])
    redis_conn.set(redis_key, 0)
    episodes_file = os.path.join(chk_dir, 'episodes_latest')
    pickle.dump(paths, open(episodes_file, 'wb'))

def get_paths_from_latest_policy(pickled_obj):
    """
    pickled_obj = (env_name, difficulty, max_obstacles, filter_type, history_len,
              chk_dir, scaler, batch_size_per_cores, batch_size)
    """
    env_name = pickled_obj[0]
    difficulty = pickled_obj[1]
    max_obstacles = pickled_obj[2]
    filter_type = pickled_obj[3]
    history_len = pickled_obj[4]
    chk_dir = pickled_obj[5]
    scaler = pickled_obj[6]
    batch_size_per_core = pickled_obj[7]
    batch_size = pickled_obj[8]
    redis_conn = redis.Redis()
    redis_key = 'curr_batch_size-' + chk_dir
    with tf.Session() as sess:
        data = joblib.load(os.path.join(chk_dir, 'params.pkl'))
        policy = data['policy']
        env = TfEnv(GymEnv(env_name, difficulty=difficulty,
            runenv_seed=1, visualize=False,
            record_log=False, record_video=False, filter_type=filter_type,
            history_len=history_len, max_obstacles=max_obstacles))
        total_length = 0
        paths = []
        # while total_length < batch_size_per_core:
        while True:
            path = rollout(env, policy, max_path_length=1000,
                animated=False, always_return_paths=True, scaler=scaler,
                redis_conn=redis_conn, redis_key=redis_key, batch_size=batch_size)
            paths.append(path)
            redis_conn.incrby(redis_key, len(path['rewards']))
            b_size = int(redis_conn.get(redis_key))
            total_length += len(path['rewards'])
            if b_size >= batch_size:
                break
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
            max_obstacles = int(query['max_obstacles'][0])
            if 'filter_type' in query:
                filter_type = str(query['filter_type'][0])
            else:
                filter_type = ''
            history_len = int(query['history_len'][0])
            dump_episodes(env_name, difficulty, chk_dir, batch_size, cores,
                          max_obstacles, filter_type, history_len)
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