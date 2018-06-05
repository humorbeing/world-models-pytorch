import gym
from PIL import Image

def generate_rollout(env_name="CarRacing-v0"):
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
        env.render()
        action = env.action_space.sample()
        # action = [-1, 1, 1]
        # print(env)
        # print(action)
        state, r, done, _ = env.step(action)
        frame = Image.fromarray(state, 'RGB')
        frames.append([action, frame])
        print('counter:', counter)
        counter += 1
        # actions.append(action[None, :])
    # actions = np.concatenate(actions)
    #     frames = np.concatenate(frames)
    # return frames, actions

generate_rollout()
