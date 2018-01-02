from sandbox.rocky.tf.launchers.launcher_utils import FLAGS, get_env_info, get_annotations_string
from sandbox.rocky.tf.launchers.launcher_stub_utils import get_env, get_policy, get_baseline, get_qf, get_es, get_algo
from rllab.misc.instrument import run_experiment_lite
from rllab import config
import os.path as osp
import sys, os
import tensorflow as tf
from copy import deepcopy
import numpy as np
from pymongo import MongoClient
from bson.objectid import ObjectId

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

MONGO_HOST = '172.30.0.169'
MONGO_PORT = 27017
MONGO_DB = 'eschernode'
mongoClient = MongoClient(MONGO_HOST, MONGO_PORT, connect=False)

db = mongoClient[MONGO_DB]

MAIN_KEYS = [('userId', 'user')]

NON_VARIANT_KEYS = {'string': ['env_name', 'exp', 'baseline_hidden_sizes', 'qf_hidden_nonlinearity',
                               'qf_hidden_sizes', 'policy_output_nonlinearity', 'policy_hidden_nonlinearity',
                               'policy_hidden_sizes', 'algo_name'], 
                    'int': ['max_episode', 'n_parallel'],
                    'float': []
                   } 

VARIANT_KEYS = {'string': [],
                'int': ['seed', 'qf_batch_size', 'policy_batch_size', 'replay_pool_size', 'batch_size'],
                'float': ['qf_learning_rate', 'scale_reward', 'step_size', 'gae_lambda', 'learning_rate', 'discount']
               }

def getExperimentById(expId):
    return db.experiments.find_one({'_id': ObjectId(expId)})

def get_exp_config(expId, variantIndex):
    exp = getExperimentById(expId)
    assert(exp != None)
    config = {}
    for item in MAIN_KEYS:
        config[item[0]] = exp[item[1]]
    for key in NON_VARIANT_KEYS['string']:
        config[key] = exp['config'][key]
    for key in NON_VARIANT_KEYS['int']:
        config[key] = int(exp['config'][key])
    for key in NON_VARIANT_KEYS['float']:
        config[key] = float(exp['config'][key])
    for key in VARIANT_KEYS['string']:
        config[key] = exp['config']['variants'][variantIndex][key]
    for key in VARIANT_KEYS['int']:
        config[key] = int(exp['config']['variants'][variantIndex][key])
    for key in VARIANT_KEYS['float']:
        config[key] = float(exp['config']['variants'][variantIndex][key])
    # Just use experiment id as experiment name for consistency
    config['exp'] = expId
    return config


def set_experiment(mode="local", keys=None, params=dict()):
    flags = FLAGS.__flags
    flags = deepcopy(flags)

    if flags['expId'] == 'expId':
        experiment_config_from_mongo = {}
    else:
        print("Getting exp config from mongo")
        variantIndex = int(flags['variantId'])
        expId = flags['expId']
        experiment_config_from_mongo = get_exp_config(expId, variantIndex)
    
    for k, v in params.items():
        print('Modifying flags.%s from %r to %r'%(
            k, flags[k], v))
        flags[k] = v

    for k, v in experiment_config_from_mongo.items():
        print('Getting flags from mongo. %s from %r to %r'%(
            k, flags[k], v))
        flags[k] = v

    n_episodes = flags["max_episode"] # max episodes before termination
    info, _ = get_env_info(**flags)
    max_path_length = info['horizon']
    n_itr = int(np.ceil(float(n_episodes*max_path_length)/flags['batch_size']))
    print('No of iteration are %s'%n_itr)

    exp_prefix='%s'%(flags["exp"])
    exp_name=get_annotations_string(keys=keys, **flags)
    if flags["normalize_obs"]: flags["env_name"] += 'norm'
    exp_name = '%s-%d--'%(flags["env_name"], flags["batch_size"]) + exp_name
    log_dir = config.LOG_DIR + "/local/" + exp_prefix.replace("_", "-") + "/" + exp_name
    if flags["seed"] is not None:
        log_dir += '--s-%d'%flags["seed"]

    # over writing log_dir to be exp_checkpoint
    log_dir = config.LOG_DIR + "/local/" + "exp_data"

    if not flags["overwrite"] and osp.exists(log_dir):
        ans = input("Overwrite %s?: (yes/no)"%log_dir)
        if ans != 'yes': sys.exit(0)

    env = get_env(record_video=True, record_log=True, **flags)
    policy = get_policy(env=env, info=info, **flags)
    baseline = get_baseline(env=env, **flags)
    qf = get_qf(env=env, info=info, **flags)
    es = get_es(env=env, info=info, **flags)

    algo = get_algo(n_itr=n_itr, env=env, policy=policy, baseline=baseline,
            qf=qf, es=es, max_path_length=max_path_length, **flags)
    return algo, dict(
            exp_prefix=exp_prefix,
            exp_name=exp_name,
            mode=mode,
            seed=flags["seed"],
            n_parallel=flags["n_parallel"]
            )

def run_experiment(**kwargs):
    algo, run_kwargs = set_experiment(**kwargs)
    run_experiment_lite(
        algo.train(),
        # n_parallel=1, # added this in flags
        snapshot_mode="last_best",
        terminate_machine=True,
        sync_s3_pkl=True,
        periodic_sync_interval=1200,
        # terminate_machine=False,
        # fast_code_sync=False,
        **run_kwargs,
    )

def main(argv=None):
    print(argv)
    print("#####################################")
    run_experiment(mode="local")

if __name__ == '__main__':
    tf.app.run()
