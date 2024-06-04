"""
Module for the Actor-Critic RL protocol.
"""

import typing

import numpy as np

from swarmrl.actions.actions import Action
from swarmrl.agents.agent import Agent
from swarmrl.components.colloid import Colloid
from swarmrl.intrinsic_reward.intrinsic_reward import IntrinsicReward
from swarmrl.losses import Loss, ProximalPolicyLoss
from swarmrl.networks.network import Network
from swarmrl.observables.observable import Observable
from swarmrl.tasks.task import Task
from swarmrl.utils.colloid_utils import TrajectoryInformation

import pathlib
import h5py


class ActorCriticAgent(Agent):
    """
    Class to handle the actor-critic RL Protocol.
    """

    def __init__(
        self,
        particle_type: int,
        network: Network,
        task: Task,
        observable: Observable,
        actions: dict,
        loss: Loss = ProximalPolicyLoss(),
        train: bool = True,
        intrinsic_reward: IntrinsicReward = None,
        record_agent: bool = False,
        out_folder: str = "./Agent_Data",
    ):
        """
        Constructor for the actor-critic protocol.

        Parameters
        ----------
        particle_type : int
                Particle ID this RL protocol applies to.
        observable : Observable
                Observable for this particle type and network input
        task : Task
                Task for this particle type to perform.
        actions : dict
                Actions allowed for the particle.
        loss : Loss (default=ProximalPolicyLoss)
                Loss function to use to update the networks.
        train : bool (default=True)
                Flag to indicate if the agent is training.
        intrinsic_reward : IntrinsicReward (default=None)
                Intrinsic reward to use for the agent.
        record_agent : bool (default=False)
                Flag to indicate if the agent should record data.
        out_folder : str (default="./Agent_Data")
                Folder to store the agent data.
        """
        # Properties of the agent.
        self.network = network
        self.particle_type = particle_type
        self.task = task
        self.observable = observable
        self.actions = actions
        self.train = train
        self.loss = loss
        self.intrinsic_reward = intrinsic_reward
        self.record_agent = record_agent
        self.out_folder = pathlib.Path(out_folder)
        self.is_stored = False

        # Trajectory to be updated.
        self.trajectory = TrajectoryInformation(particle_type=self.particle_type)

    def __name__(self) -> str:
        """
        Give the class a name.

        Return
        ------
        name : str
            Name of the class.
        """
        return "ActorCriticAgent"

    def update_agent(self) -> tuple:
        """
        Update the agents network.

        Returns
        -------
        rewards : float
                Net reward for the agent.
        killed : bool
                Whether or not this agent killed the
                simulation.
        """
        # Collect data for returns.
        rewards = self.trajectory.rewards
        killed = self.trajectory.killed

        # Compute loss for actor and critic.
        self.loss.compute_loss(
            network=self.network,
            episode_data=self.trajectory,
        )

        # Update the intrinsic reward if set.
        if self.intrinsic_reward:
            self.intrinsic_reward.update(self.trajectory)



        # Save the agents data 
        if self.record_agent == True:
        
            if self.is_stored == False:
                self.h5_filename = self.out_folder / "agent_data.hdf5"
                self.out_folder.mkdir(parents=True, exist_ok=True)
                self.data_holder = {
                    'features': list(),
                    'actions': list(),
                    'log_probs': list(),
                    'rewards': list(),
                }
                n_colloids = np.array(self.trajectory.features).shape[1]
                episode_length = np.array(self.trajectory.features).shape[0]

                with h5py.File(self.h5_filename.as_posix(), 'a') as h5_outfile:
                    agent_group = h5_outfile.require_group(f"Agent_{self.particle_type}")
                    dataset_kwargs = dict(compression="gzip")
                    
                    agent_group.require_dataset(
                        'features',
                        shape = (1, episode_length, n_colloids, 1),
                        maxshape=(None, episode_length, n_colloids, 1),
                        dtype = np.float32,
                        **dataset_kwargs,
                    )
                    for name in ['actions', 'log_probs', 'rewards']:
                        agent_group.require_dataset(
                            name,
                            shape=(1, episode_length, n_colloids),
                            maxshape=(None, episode_length, n_colloids),
                            dtype = int if name == 'actions' else float,
                            **dataset_kwargs,
                        )

                self.is_stored = True

                self.write_idx = 0
                self.h5_time_steps_written = 0

            # Append the data
            n_new_timesteps = 1 # because only after one episode this gets called

            self.data_holder['features'].append(self.trajectory.features)
            self.data_holder['actions'].append(self.trajectory.actions)
            self.data_holder['log_probs'].append(self.trajectory.log_probs)
            self.data_holder['rewards'].append(self.trajectory.rewards)


            with h5py.File(self.h5_filename.as_posix(), 'a') as h5_outfile:
                agent_group = h5_outfile[f"Agent_{self.particle_type}"]

                for key in self.data_holder.keys():

                    dataset = agent_group[key]
                    values = np.stack(self.data_holder[key], axis=0)
                    dataset.resize(self.write_idx + values.shape[0], axis = 0)
                    dataset[self.write_idx : self.write_idx + values.shape[0]] = values
            
            self.h5_time_steps_written =+ n_new_timesteps


        # Reset the trajectory storage.
        self.reset_trajectory()

        return rewards, killed

    def reset_agent(self, colloids: typing.List[Colloid]):
        """
        Reset several properties of the agent.

        Reset the observables and tasks for the agent.

        Parameters
        ----------
        colloids : typing.List[Colloid]
                Colloids to use in the initialization.
        """
        self.observable.initialize(colloids)
        self.task.initialize(colloids)
        # TODO Regard agent storing?

    def reset_trajectory(self):
        """
        Set all trajectory data to None.
        """
        self.task.kill_switch = False  # Reset here.
        self.trajectory = TrajectoryInformation(particle_type=self.particle_type)

    def initialize_network(self):
        """
        Initialize all of the models in the gym.
        """
        self.network.reinitialize_network()

    def save_agent(self, directory: str):
        """
        Save the agent network state.

        Parameters
        ----------
        directory : str
                Location to save the models.
        """
        self.network.export_model(
            filename=f"{self.__name__()}_{self.particle_type}", directory=directory
        )

    def restore_agent(self, directory: str):
        """
        Restore the agent state from a directory.
        """
        self.network.restore_model_state(
            filename=f"{self.__name__()}_{self.particle_type}", directory=directory
        )

    def calc_action(self, colloids: typing.List[Colloid]) -> typing.List[Action]:
        """
        Copmute the new state for the agent.

        Returns the chosen actions to the force function which
        talks to the espresso engine.

        Parameters
        ----------
        colloids : List[Colloid]
                List of colloids in the system.
        """
        state_description = self.observable.compute_observable(colloids)
        action_indices, log_probs = self.network.compute_action(
            observables=state_description
        )
        chosen_actions = np.take(list(self.actions.values()), action_indices, axis=-1)

        # Compute extrinsic rewards.
        rewards = self.task(colloids)
        # Compute intrinsic rewards if set.
        if self.intrinsic_reward:
            rewards += self.intrinsic_reward.compute_reward(
                episode_data=self.trajectory
            )

        # Update the trajectory information.
        if self.train:
            self.trajectory.features.append(state_description)
            self.trajectory.actions.append(action_indices)
            self.trajectory.log_probs.append(log_probs)
            self.trajectory.rewards.append(rewards)
            self.trajectory.killed = self.task.kill_switch

        self.kill_switch = self.task.kill_switch

        return chosen_actions
