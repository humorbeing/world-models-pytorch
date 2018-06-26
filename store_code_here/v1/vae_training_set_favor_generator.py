import gym
from PIL import Image
import numpy as np
from gym.envs.box2d.car_dynamics import Car
from gym.envs.box2d import CarRacing
from scipy.misc import imresize as resize


dst = '/media/ray/SSD/workspace/python/dataset/save_here'
MAX_GAME_TIME = 1000
MAX_RUNS = 20
name_this = 'rollout_v4'
def generate_action():
    action = [0, 0, 0]
    action[0] = np.random.normal(0, np.random.random())
    acc_chance = 0.9
    if np.random.random() < acc_chance:
        action[1] = np.random.random()
        action[2] = 0
    else:
        action[1] = 0
        action[2] = np.random.random()
    return action


def _process_frame(frame):
    obs = frame[0:84, :, :].astype(np.float)/255.0
    obs = resize(obs, (64, 64))
    obs = ((1.0 - obs) * 255).round().astype(np.uint8)
    # obs = (obs * 255).round().astype(np.uint8)
    return obs


# REST_NUM = 250

import matplotlib.pyplot as plt
def multiple_runs():
    env = CarRacing()

    states = []
    actions = []
    for run in range(MAX_RUNS):
        state = env.reset()
        env.render() # must have!
        # done = False
        counter = 0
        for game_time in range(MAX_GAME_TIME):
            # env.render()
            action = generate_action()
            state = _process_frame(state)
            # if game_time == 5:
            #     plt.imshow(state)
            #     plt.show()
            #     state = _process_frame(state)
            #     plt.imshow(state)
            #     plt.show()
            states.append(state)
            actions.append(action)
            state, r, done, _ = env.step(action)

            # print(r)
            print(
                'RUN:{},GT:{},DATA:{}'.format(
                    run, game_time, len(states)
                )
            )
            # if counter == REST_NUM:
            #
            #     position = np.random.randint(len(env.track))
            #     env.car = Car(env.world, *env.track[position][1:4])
            #     counter = 0
            # counter += 1
        states = np.array(states, dtype=np.uint8)
        actions = np.array(actions, dtype=np.float16)
        save_name = name_this + '_{}.npz'.format(run)
        # np.save(dst + '/' + save_name, frame_and_action)

        np.savez_compressed(dst + '/' + save_name, action=actions, state=states)
        states = []
        actions = []



multiple_runs()








