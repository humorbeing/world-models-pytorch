import gym
from PIL import Image
import numpy as np
from gym.envs.box2d.car_dynamics import Car
from gym.envs.box2d import CarRacing

dst = '/media/ray/SSD/workspace/python/dataset/save_here'

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



MAX_GAME_TIME = 1000
MAX_RUNS = 20
REST_NUM = 250
# actions = []
def multiple_runs(on):
    env = CarRacing()
    frame_and_action = []
    for run in range(MAX_RUNS):
        env.reset()
        # done = False
        counter = 0
        for game_time in range(MAX_GAME_TIME):
            # env.render()
            action = generate_action()
            state, r, done, _ = env.step(action)
            frame_and_action.append({'state': state,
                           'action': action})
            # print(r)
            counter += 1
            if counter > REST_NUM:
                print(
                    'RUN:{},GT:{},DATA:{}'.format(
                        run, game_time, len(frame_and_action)
                    )
                )
                position = np.random.randint(len(env.track))
                env.car = Car(env.world, *env.track[position][1:4])
                counter = 0
    save_name = 'rollout_{}.npy'.format(on)
    np.save(dst + '/' + save_name, frame_and_action)


for i in range(1, 2):
    multiple_runs(i)

# print(frame)
# print(state)
# print(state.shape)
# print(state[10:20, 10:20, 2])
# frames = np.array(frames)
# print(len(frames))
# print(frames.shape)







# for _ in range(10):
#     print(generate_action())