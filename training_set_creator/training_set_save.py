import gym
from PIL import Image
import numpy as np


env_name="CarRacing-v0"
env = gym.make(env_name)
frames = []

actions = []
state = env.reset()
# frame = Image.fromarray(state, 'RGB')
# frames.append(frame)
# s = env.render("rgb_array")
# print(s)
# print(env.action_space)
# a = env.action_space.sample()
# print(a.shape)
# print(a[None, :].shape)
done = False
# while True:
#     a = env.action_space.sample()
#     print(a[None, :])
counter = 0
while not done:
# while counter < 200:
    env.render()
    action = env.action_space.sample()
    # action = [-1, 1, 1]
    # print(env)
    # print(action)
    state, r, done, _ = env.step(action)
    # frame = Image.fromarray(state, 'RGB')

    frames.append({'state': state,
                   'action': action})
    print('counter:', counter)
    counter += 1
    # break
    # actions.append(action[None, :])
# actions = np.concatenate(actions)
#     frames = np.concatenate(frames)
# return frames, actionsenv_name="CarRacing-v0"

# print(frame)
# print(state)
# print(state.shape)
# print(state[10:20, 10:20, 2])
frames = np.array(frames)
print(len(frames))
print(frames.shape)
dst = '/media/ray/SSD/workspace/python/dataset/save_here'
save_name = '2.npy'
np.save(dst + '/' + save_name, frames)