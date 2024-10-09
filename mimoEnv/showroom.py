""" Simple script to view the showroom. We perform no training and MIMo takes no actions.
"""

import gymnasium as gym
import time
import numpy as np
import mimoEnv

SITTING_POSITION = {
    "robot:hip_lean1": np.array([0.039088]), "robot:hip_rot1": np.array([0.113112]),
    "robot:hip_bend1": np.array([0.5323]), "robot:hip_lean2": np.array([0]), "robot:hip_rot2": np.array([0]),
    "robot:hip_bend2": np.array([0.5323]),
    "robot:head_swivel": np.array([0]), "robot:head_tilt": np.array([0]), "robot:head_tilt_side": np.array([0]),
    "robot:left_eye_horizontal": np.array([0]), "robot:left_eye_vertical": np.array([0]),
    "robot:left_eye_torsional": np.array([0]), "robot:right_eye_horizontal": np.array([0]),
    "robot:right_eye_vertical": np.array([0]), "robot:right_eye_torsional": np.array([0]),
    "robot:left_shoulder_horizontal": np.array([0.683242]), "robot:left_shoulder_ad_ab": np.array([0.3747]),
    "robot:left_shoulder_rotation": np.array([-0.62714]), "robot:left_elbow": np.array([-0.756016]),
    "robot:left_hand1": np.array([0.28278]), "robot:left_hand2": np.array([0]), "robot:left_hand3": np.array([0]),
    "robot:right_hip1": np.array([-1.51997]), "robot:right_hip2": np.array([-0.397578]),
    "robot:right_hip3": np.array([0.0976615]), "robot:right_knee": np.array([-1.85479]),
    "robot:right_foot1": np.array([-0.585865]), "robot:right_foot2": np.array([-0.358165]),
    "robot:right_foot3": np.array([0]), "robot:right_toes": np.array([0]),
    "robot:left_hip1": np.array([-1.23961]), "robot:left_hip2": np.array([-0.8901]),
    "robot:left_hip3": np.array([0.7156]), "robot:left_knee": np.array([-2.531]),
    "robot:left_foot1": np.array([-0.63562]), "robot:left_foot2": np.array([0.5411]),
    "robot:left_foot3": np.array([0.366514]), "robot:left_toes": np.array([0.24424]),
}
def main():
    """ Creates the environment and takes 200 time steps. MIMo takes no actions.
    The environment is rendered to an interactive window.
    """

    env = gym.make("MIMoShowroom-v0", show_sensors=False, print_space_sizes=True, initial_qpos=SITTING_POSITION)

    max_steps = 200

    _ = env.reset()

    start = time.time()
    for step in range(max_steps):
        action = np.zeros(env.action_space.shape)
        obs, reward, done, trunc, info = env.step(action)
        env.render()
        if done or trunc:
            env.reset()

    print("Elapsed time: ", time.time() - start, "Simulation time:", max_steps*env.dt)
    env.close()


if __name__ == "__main__":
    main()
