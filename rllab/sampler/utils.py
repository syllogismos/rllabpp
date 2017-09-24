import numpy as np
from rllab.misc import tensor_utils
import time, redis

def rollout(env, agent, max_path_length=np.inf, animated=False, speedup=1,
            always_return_paths=False, scaler=None, redis_conn=None,
            redis_key='None', batch_size=1000000):
    unscaled_obs = []
    observations = []
    actions = []
    rewards = []
    terminals = []
    agent_infos = []
    env_infos = []
    o = env.reset()
    agent.reset()
    path_length = 0
    if scaler is not None:
        scale, offset = scaler.get()
    else:
        scale = [1.0]*env.observation_space.shape[0]
        offset = [0.0]*env.observation_space.shape[0]
    if animated:
        env.render()
    while path_length < max_path_length:
        o = env.observation_space.flatten(o)
        unscaled_obs.append(o)
        obs = (o - offset) * scale
        observations.append(obs)
        a, agent_info = agent.get_action(obs)
        next_o, r, d, env_info = env.step(a)
        rewards.append(r)
        terminals.append(d)
        actions.append(env.action_space.flatten(a))
        agent_infos.append(agent_info)
        env_infos.append(env_info)
        path_length += 1
        if d:
            break
        o = next_o
        if animated:
            env.render()
            timestep = 0.05
            time.sleep(timestep / speedup)
        if redis_conn is not None:
            try:
                curr_batch_size = redis_conn.get(redis_key)
                if curr_batch_size is None:
                    curr_batch_size = 0
                curr_batch_size = int(curr_batch_size)
                if curr_batch_size > batch_size:
                    break
            except:
                print("redis get failed while rollout")
                pass
            
    if animated and not always_return_paths:
        return

    return dict(
        unscaled_obs=tensor_utils.stack_tensor_list(unscaled_obs),
        observations=tensor_utils.stack_tensor_list(observations),
        actions=tensor_utils.stack_tensor_list(actions),
        rewards=tensor_utils.stack_tensor_list(rewards),
        terminals=tensor_utils.stack_tensor_list(terminals),
        agent_infos=tensor_utils.stack_tensor_dict_list(agent_infos),
        env_infos=tensor_utils.stack_tensor_dict_list(env_infos),
    )
