
import gym, roboschool, json

a = list(gym.envs.registry.all())

l = list(map(lambda x: x.id, a))

parameter = ['CNNClassifierTraining-v0', 'ConvergenceControl-v0']
box2d = ['LunarLander-v2', 'LunarLanderContinuous-v2', 'BipedalWalker-v2', 'BipedalWalkerHardcore-v2', 'CarRacing-v0']
mujoco = ['Reacher-v1',
 'Pusher-v0',
 'Thrower-v0',
 'Striker-v0',
 'InvertedPendulum-v1',
 'InvertedDoublePendulum-v1',
 'HalfCheetah-v1',
 'Hopper-v1',
 'Swimmer-v1',
 'Walker2d-v1',
 'Ant-v1',
 'Humanoid-v1',
 'HumanoidStandup-v1']

working = list(set(l) - set(mujoco + parameter + box2d))

def get_data(env):
    d = {}
    d['id'] = env.spec.id
    d['observation_space'] = str(env.observation_space)
    d['action_space'] = str(env.action_space)
    d['timestep_limit'] = ''
    d['max_episode_steps'] = ''
    d['reward_threshold'] = ''
    if hasattr(env.spec, 'timestep_limit'):
        d['timestep_limit'] = env.spec.timestep_limit
    if hasattr(env.spec, 'max_episode_steps'):
        d['max_episode_steps'] = env.spec.max_episode_steps
    if hasattr(env.spec, 'reward_threshold'):
        d['reward_threshold'] = env.spec.reward_threshold
    return d

meta_data = []

for i, e in enumerate(working):
    print(i, e)
    env = gym.make(e)
    meta_data.append(get_data(env))


json.dump(meta_data, open('env_meta_data.json', 'w'))