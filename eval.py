import numpy as np
import h5py

import gymnasium as gym
from gymnasium.wrappers.pixel_observation import PixelObservationWrapper
from tactile_envs.utils.resize_dict import ResizeDict
from tactile_envs.utils.add_tactile import AddTactile
from utils.frame_stack import FrameStack

from models.ppo_mae import PPO_MAE


env_name = "HandManipulateBlockRotateZFixed-v1"
state_type = "vision_and_touch"
frame_stack = 4

env = gym.make(env_name, render_mode="rgb_array", reward_type='dense')
env = PixelObservationWrapper(env, pixels_only=False)
env = ResizeDict(env, 64, pixel_key='pixels')
if state_type == "vision_and_touch":
    env = AddTactile(env)
env = FrameStack(env, frame_stack)

model = PPO_MAE.load("logs/rl_model_3000000_steps")

is_done = True
is_success = False
num_saved_traj = 0
num_max_traj = 1082

with h5py.File(f"expert_demo_block_{num_max_traj}_1.hdf5", "w") as f:
    obs_dataset = f.create_dataset("observations", shape=(0, 61), maxshape=(None, 61), dtype=np.float32)
    action_dataset = f.create_dataset("actions", shape=(0, 20), maxshape=(None, 20), dtype=np.float32)
    reward_dataset = f.create_dataset("rewards", shape=(0,), maxshape=(None,), dtype=np.float32)
    terminated_dataset = f.create_dataset("terminals", shape=(0,), maxshape=(None,), dtype=np.float32)
    is_success_dataset = f.create_dataset("is_successes", shape=(0,), maxshape=(None,), dtype=np.float32)
    truncated_dataset = f.create_dataset("truncates", shape=(0,), maxshape=(None,), dtype=np.float32)
    info_dtype = np.dtype([
        ("qpos", np.float32, 38),
        ("qvel", np.float32, 36),
        ("achieved_goal", np.float32, 7),
        ("desired_goal", np.float32, 7)
    ])
    info_dataset = f.create_dataset("infos", shape=(0,), maxshape=(None,), dtype=info_dtype)

    while num_saved_traj < num_max_traj:
        if is_done:
            if is_success:
                #  np.save(f"image_obses-{num_saved_traj}.npy", np.array(obses))
                #  print("success traj saved!")
                observations = np.array(observations)
                actions = np.array(actions)
                rewards = np.array(rewards)
                terminals = np.array(terminals)
                qposes = np.array(qposes)
                qvels = np.array(qvels)
                desired_goals = np.array(desired_goals)
                achieved_goals = np.array(achieved_goals)
                is_successes = np.array(is_successes)
                truncates = np.array(truncates)

                obs_dataset.resize(obs_dataset.shape[0] + observations.shape[0], axis=0)
                obs_dataset[-observations.shape[0]:] = observations

                action_dataset.resize(action_dataset.shape[0] + actions.shape[0], axis=0)
                action_dataset[-actions.shape[0]:] = actions

                reward_dataset.resize(reward_dataset.shape[0] + rewards.shape[0], axis=0)
                reward_dataset[-rewards.shape[0]:] = rewards

                terminated_dataset.resize(terminated_dataset.shape[0] + terminals.shape[0], axis=0)
                terminated_dataset[-terminals.shape[0]:] = terminals

                is_success_dataset.resize(is_success_dataset.shape[0] + is_successes.shape[0], axis=0)
                is_success_dataset[-is_successes.shape[0]:] = is_successes

                truncated_dataset.resize(terminated_dataset.shape[0] + truncates.shape[0], axis=0)
                terminated_dataset[-truncates.shape[0]:] = truncates

                infos = np.zeros(len(qposes), dtype=info_dtype)
                infos['qpos'] = qposes
                infos['qvel'] = qvels
                infos['desired_goal'] = desired_goals
                infos['achieved_goal'] = achieved_goals
                info_dataset.resize(info_dataset.shape[0] + infos.shape[0], axis=0)
                info_dataset[-infos.shape[0]:] = infos

                print(f"expert data saved: {num_saved_traj}")
                num_saved_traj += 1

            obs = env.reset()
            is_done = False
            observations, actions, rewards, terminals, qposes, qvels, desired_goals, achieved_goals, is_successes, truncates = [], [], [], [], [], [], [], [], [], []

        if isinstance(obs, tuple):
            obs = obs[0]
        
        observations.append(obs["observation"][-1])
        desired_goals.append(obs["desired_goal"][-1])
        achieved_goals.append(obs["achieved_goal"][-1])
        qposes.append(env.unwrapped.data.qpos)
        qvels.append(env.unwrapped.data.qvel)

        _obs = {"image": obs["pixels"]} | {"tactile": obs["tactile"]}
        action, _states = model.predict(_obs, deterministic=True)
        next_obs, reward, terminated, truncated, info = env.step(action)  # info: {'is_success': 0.0}
        
        actions.append(action)
        rewards.append(reward)
        terminals.append(terminated)
        truncates.append(truncated)
        is_successes.append(info["is_success"])

        is_done = terminated or truncated
        is_success = (info["is_success"] == 1.0)
        obs = next_obs
