import gym
from PIL import Image
import torch

def generate_rollout(env_name="CarRacing-v0"):
    env = gym.make(env_name)
    frames = []
    actions = []

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
    while True:
        state = env.reset()
        for _ in range(150):
            env.render()
            action = env.action_space.sample()
            # action = [-1, 1, 1]
            # print(env)
            # print(action)
            state, r, done, _ = env.step(action)
        # frame = Image.fromarray(state, 'RGB')
        # frames.append([action, frame])
        # print('counter:', counter)
        # counter += 1
        # break
        # actions.append(action[None, :])
    # actions = np.concatenate(actions)
    #     frames = np.concatenate(frames)
    # return frames, actions

generate_rollout()


# a = torch.rand(5,6,3)
# b = torch.rand(5,6,20)
# print(a.shape)
# print(b.shape)
# # c = torch.cat((a, b), dim=-1)
# c = torch.cat((a, b), dim=2)
# print(c.shape)
# import numpy as np
# a = np.random.random(3)
# print(a)