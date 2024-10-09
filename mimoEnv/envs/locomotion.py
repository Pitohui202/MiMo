""" This module contains a simple experiment where MIMo is tasked with touching parts of his own body.

The scene is empty except for MIMo, who is sitting on the ground. The task is for MIMo to touch a randomized target
body part with his right arm. MIMo is fixed in the initial sitting position and can only move his right arm.
Sensory inputs consist of touch and proprioception. Proprioception uses the default settings, but touch excludes
several body parts and uses a lowered resolution to improve runtime.
The body part can be any of the geoms constituting MIMo.

MIMos initial position is constant in all episodes. The target body part is randomized. An episode is completed
successfully if MIMo touches the target body part with his right arm.

The reward structure consists of a large fixed reward for touching the right body part, a shaping reward for touching
another body part, depending on the distance between the contact and the target body part, and a penalty for each time
step.

The class with the environment is :class:`~mimoEnv.envs.selfbody.MIMoSelfBodyEnv` while the path to the scene XML is
defined in :data:`SELFBODY_XML`.
"""
import csv
import os
import random

import numpy
import numpy as np

from mimoEnv.envs.mimo_env import MIMoEnv, DEFAULT_PROPRIOCEPTION_PARAMS, SCENE_DIRECTORY
import mimoEnv.utils as env_utils
from mimoActuation.actuation import SpringDamperModel
from mimoActuation.muscle import MuscleModel





SITTING_POSITION = {
    "robot:hip_bend1": np.array([0.5823]),
    "robot:hip_bend2": np.array([0.5823]),
    "robot:head_swivel": np.array([0]), "robot:head_tilt": np.array([0]), "robot:head_tilt_side": np.array([0]),
    "robot:left_eye_horizontal": np.array([0]), "robot:left_eye_vertical": np.array([0]),
    "robot:left_eye_torsional": np.array([0]), "robot:right_eye_horizontal": np.array([0]),
    "robot:right_eye_vertical": np.array([0]), "robot:right_eye_torsional": np.array([0]),
    "robot:left_shoulder_horizontal": np.array([0]), "robot:left_shoulder_ad_ab": np.array([0]),
    "robot:left_shoulder_rotation": np.array([0.778]), "robot:left_elbow": np.array([0]),
    "robot:left_hand1": np.array([1.57]), "robot:left_hand2": np.array([1.5]), "robot:left_hand3": np.array([-0.925]),
    "robot:right_hip1": np.array([-1.16]), "robot:right_hip2": np.array([-0.184]),
    "robot:right_hip3": np.array([0]), "robot:right_knee": np.array([-1.1]),
    "robot:right_foot1": np.array([-1]), "robot:right_foot2": np.array([0]),
    "robot:right_foot3": np.array([0]), "robot:right_toes": np.array([0]),
    "robot:left_hip1": np.array([-1.16]), "robot:left_hip2": np.array([-0.184]),
    "robot:left_hip3": np.array([0]), "robot:left_knee": np.array([-1.1]),
    "robot:left_foot1": np.array([-1]), "robot:left_foot2": np.array([0]),
    "robot:left_foot3": np.array([0]), "robot:left_toes": np.array([0]),
}
""" Initial position of MIMo. Specifies initial values for all joints.
We grabbed these values by posing MIMo using the MuJoCo simulate executable and the positional actuator file.
We need these not just for the initial position but also resetting the position (excluding the right arm) each step.

:meta hide-value:
"""


LOCOMOTION_XML = os.path.join(SCENE_DIRECTORY, "locomotion_scene.xml")
""" Path to the scene for this experiment.

:meta hide-value:
"""


class MIMoLocomotionEnv(MIMoEnv):
    """ MIMo learns about his own body.

    MIMo is tasked with touching a given part of his body using his right arm.
    Attributes and parameters are mostly identical to the base class, but there are two changes.
    The constructor takes two arguments less, `goals_in_observation` and `done_active`, which are both permanently
    set to ``True``.
    Finally, there are two extra attributes for handling the goal state. The :attr:`.goal` attribute stores the target
    geom in a one hot encoding, while :attr:`.target_geom` and :attr:`.target_body` store the geom and its associated
    body as an index. For more information on geoms and bodies please see the MuJoCo documentation.

    Attributes:
        target_geom (int): The body part MIMo should try to touch, as a MuJoCo geom.
        target_body (str): The name of the kinematic body that the target geom is a part of.
        init_sitting_qpos (numpy.ndarray): The initial position.
    """

    def __init__(self,
                 model_path=LOCOMOTION_XML,
                 initial_qpos=SITTING_POSITION,
                 frame_skip=1,
                 proprio_params=DEFAULT_PROPRIOCEPTION_PARAMS,
                 vision_params=None,
                 vestibular_params=None,
                 actuation_model=SpringDamperModel,
                 goals_in_observation=False,
                 done_active=True,
                 **kwargs,
                 ):


        super().__init__(model_path=model_path,
                         initial_qpos=initial_qpos,
                         frame_skip=frame_skip,
                         proprio_params=proprio_params,
                         vision_params=vision_params,
                         vestibular_params=vestibular_params,
                         actuation_model=actuation_model,
                         goals_in_observation=goals_in_observation,
                         done_active=done_active,
                         **kwargs)
        self.reward_steps=0
        self.reward_sum=0
        self.head_sum=0
        self.square_sum=0
        self.hlplist1=[]
        self.hlplist2=[]
        env_utils.set_joint_qpos(self.model,
                                 self.data,
                                 "mimo_location",
                                 np.array([0.0579584, -0.00157173, 0.0566738, 0.892294, -0.0284863, -0.450353, -0.0135029]))
        #  "mimo_location": np.array([0.0579584, -0.00157173, 0.0566738, 0.892294, -0.0284863, -0.450353, -0.0135029]),
        #for joint_name in SITTING_POSITION:
           # env_utils.lock_joint(self.model, joint_name, joint_angle=SITTING_POSITION[joint_name][0])
        # Let sim settle for a few timesteps to allow weld and locks to settle
        self.do_simulation(np.zeros(self.action_space.shape), 25)
        self.init_sitting_qpos = self.data.qpos.copy()
        self.episodes=-1
        self.secrew=False


    def sample_goal(self):
        """Samples a new goal and returns it.

        The goal consists of a target geom that we try to touch, returned as a one-hot encoding.
        We also populate :attr:`.target_geom` and :attr:`.target_body`. which are used by other functions.

        Returns:
            numpy.ndarray: The target geom in a one hot encoding.
        """
        # randomly select geom as target (except for 2 latest geoms that correspond to fingers)
        return 0

    def is_success(self, achieved_goal, desired_goal):
        """ We have succeeded when we have a touch sensation on the goal body.

        We ignore the :attr:`.goal` attribute in this for performance reasons and determine the success condition
        using :attr:`.target_geom` instead. This allows us to save a number of array operations each step.

        Args:
            achieved_goal (object): This parameter is ignored.
            desired_goal (object): This parameter is ignored.

        Returns:
            bool: If MIMo has touched the target geom.
        """
        # check if contact with target geom:
        return 0

    def compute_reward(self, achieved_goal, desired_goal, info):
        """ Computes the reward each step.

        Three different rewards can be returned:

        - If we touched the target geom, the reward is 500.
        - If we touched a geom, but not the target, the reward is the negative of the distance between the touch
          contact and the target body.
        - Otherwise the reward is -1.

        Args:
            achieved_goal (object): This parameter is ignored.
            desired_goal (object): This parameter is ignored.
            info (dict): This parameter is ignored.

        Returns:
            float: The reward as described above.
        """
        bonus=0
        if self.data.body('head').xpos[2]>0.3:
            bonus=2
        reward=self.data.body('head').xpos[2]*10-0.07*np.square(self.data.ctrl).sum()+bonus
        self.reward_sum+=reward
        #print(self.data.body('head').xpos[2])
        #print(np.square(self.data.ctrl).sum())
        self.reward_steps+=1
        return reward

    def reset_model(self):
        """ Reset to the initial sitting position.

        Returns:
            Dict: Observations after reset.
        """
        # set qpos as new initial position and velocity as zero
        qpos = self.init_sitting_qpos
        qvel = np.zeros(self.data.qvel.shape)
        #with open("reward_len_data.csv","w") as file:
            #file.truncate()
            #writer=csv.writer(file)
            #try:
                #self.hlplist1.append(self.reward_steps)
            #except:
                #self.hlplist1=self.hlplist1
            #writer.writerow(self.hlplist1)
        #with open("reward_avg_data.csv","w") as file:
            #file.truncate()
            #writer=csv.writer(file)
            #try:
                #self.hlplist2.append(self.reward_sum/self.reward_steps)
            #except:
                #self.hlplist2=self.hlplist2
            #writer.writerow(self.hlplist2)
        with open("reward_long_data.csv","w") as file:
            file.truncate()
            writer=csv.writer(file)
            try:
                self.hlplist1.append(self.reward_sum/self.reward_steps)
            except:
                self.hlplist1=self.hlplist1
            writer.writerow(self.hlplist1)
            self.reward_sum=0
            self.reward_steps=0
        #with open("square_sum_data.csv","w") as file:
            #file.truncate()
            #writer=csv.writer(file)
            #try:
                #self.hlplist2.append(self.square_sum/self.reward_steps)
            #except:
                #self.hlplist2=self.hlplist2*/
        self.episodes+=1
        self.set_state(qpos, qvel)
        self.reward_steps=0
        self.reward_sum=0
        self.head_sum=0
        self.square_sum=0
        return self._get_obs()

    def is_failure(self, achieved_goal, desired_goal):
        #failure=(self.data.body('head').xpos[2]<0.3)
        return False

    def is_truncated(self):
        """ Dummy function. Always returns ``False``.

        Returns:
            bool: ``False``.
        """
        return False

    def get_achieved_goal(self):
        """ Dummy function that returns an empty array.

        Returns:
            numpy.ndarray: An empty array.
        """
        return numpy.ndarray
