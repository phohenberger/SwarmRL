"""
Training example for discrete lattice with neural network agents.

This script demonstrates how to train 5 agents on a 100x100 discrete lattice
to reach a random target using reinforcement learning.
"""

import pathlib
import numpy as np
import json
from datetime import datetime
from typing import List

import jax.numpy as jnp
import flax.linen as nn
import optax

import swarmrl as srl
from swarmrl.engine import DiscreteLatticeEngine
from swarmrl.components import Colloid
from swarmrl.actions import Action
from swarmrl.force_functions import ForceFunction


# Simulation parameters (reduced for debugging)
sim_params = {
    'FILENAME': str(datetime.now().strftime('%m-%d_%H-%M-%S')) + '_lattice',
    'LATTICE_SIZE': 100,  # Smaller lattice for debugging
    'N_AGENTS': 10,       # Fewer agents for debugging
    'N_EPISODES': 500,     # Moderate episodes for testing
    'RESET_FREQ': 10,    # Reset every 20 episodes for semi-episodic training
    'EPISODE_LENGTH': 20,  # Very short episodes for debugging movement
    'SEED': np.random.randint(1,6600),          # Fixed seed for reproducibility
    'TARGET_RADIUS': 5,  # Agents get reward if within this distance of target
}


class LatticeTargetSearchTask(srl.tasks.Task):
    """
    Task for agents to reach a random target on the discrete lattice.
    """
    
    def __init__(
        self,
        lattice_size: int = 100,
        target_radius: float = 5.0,
        reward_scale_factor: float = 10.0,
        particle_type: int = 0,
        sim_params: dict = None,
    ):
        """
        Initialize the target search task.
        
        Parameters
        ----------
        lattice_size : int
            Size of the square lattice
        target_radius : float
            Radius within which agents get reward
        reward_scale_factor : float
            Scale factor for rewards
        particle_type : int
            Type of particles to track
        sim_params : dict
            Simulation parameters
        """
        super().__init__(particle_type=particle_type)
        self.lattice_size = lattice_size
        self.target_radius = target_radius
        self.reward_scale_factor = reward_scale_factor
        self.sim_params = sim_params
        
        # Initialize random target
        self.rng = np.random.default_rng(sim_params.get('SEED', 42))
        self.target_position = self._generate_random_target()
        
    def _generate_random_target(self):
        """Generate a random target position on the lattice."""
        # Use smaller margin or no margin for small lattices
        margin = min(3, self.lattice_size // 4)
        x = self.rng.integers(margin, max(margin + 1, self.lattice_size - margin))
        y = self.rng.integers(margin, max(margin + 1, self.lattice_size - margin))
        return np.array([50, 50]) #TODO hack
    
    def reset_target(self):
        """Generate a new random target position."""
        self.target_position = self._generate_random_target()
        print(f"New target: {self.target_position}")
    
    def compute_colloid_reward(self, index: int, colloids: List[Colloid]) -> float:
        """
        Compute reward for a single agent using normalized 1/r distance-based reward.
        
        Parameters
        ----------
        index : int
            Index of the agent
        colloids : List[Colloid]
            List of all agents
            
        Returns
        -------
        float
            Reward for the agent
        """
        # Get current position
        position = np.array(colloids[index].pos[:2])  # Only x, y coordinates
        
        # Calculate distance to target
        distance_to_target = np.linalg.norm(position - self.target_position)
        
        # Normalize distance by the maximum possible distance in the lattice
        # Maximum distance is diagonal of the lattice: sqrt(2) * lattice_size
        max_distance = np.sqrt(2) * self.lattice_size
        normalized_distance = distance_to_target * 10 / max_distance # magic shift factor
        
        # Reward that goes to zero at max distance: (1 - normalized_distance) / (normalized_distance + epsilon)
        # This gives: reward = 0 when normalized_distance = 1 (max distance)
        #            reward = high when normalized_distance = 0 (zero distance)
        epsilon = 0.01  # Small value to avoid division by zero
        effective_distance = normalized_distance + epsilon
        
        reward = self.reward_scale_factor * (1 / effective_distance - 4/10)
        
        return reward
        #return float(np.clip(reward, 0.0, np.inf))  # Clip to positive rewards only
    
    def __call__(self, colloids: List[Colloid]):
        """
        Compute rewards for all agents using pure 1/r distance-based reward.
        
        Parameters
        ----------
        colloids : List[Colloid]
            List of all agents
            
        Returns
        -------
        np.ndarray
            Rewards for each agent
        """
        # Get indices of relevant agents
        colloid_indices = self.get_colloid_indices(colloids)
        
        # Calculate rewards based purely on current distance to target
        rewards = np.array([
            self.compute_colloid_reward(index, colloids) 
            for index in colloid_indices
        ])
        
        return rewards


class LatticePositionObservable(srl.observables.Observable):
    """
    Observable that provides position information and distance to target.
    """
    
    def __init__(
        self,
        lattice_size: int = 100,
        particle_type: int = 0,
    ):
        """
        Initialize the position observable.
        
        Parameters
        ----------
        lattice_size : int
            Size of the lattice
        particle_type : int
            Type of particles to observe
        """
        super().__init__(particle_type=particle_type)
        self.lattice_size = lattice_size
        self.target_position = np.array([50, 50])  # Will be updated by task
        
    def set_target_position(self, target_position: np.ndarray):
        """Set the target position for observations."""
        self.target_position = target_position
    
    def compute_single_observable(self, index: int, colloids: List[Colloid]) -> np.ndarray:
        """
        Compute observable for a single agent.
        
        Parameters
        ----------
        index : int
            Index of the agent
        colloids : List[Colloid]
            List of all agents
            
        Returns
        -------
        np.ndarray
            Observable vector for the agent
        """
        # Get agent position
        position = np.array(colloids[index].pos[:2])
        
        # Normalize position to [0, 1]
        norm_position = position / self.lattice_size
        
        # Calculate relative position to target
        relative_to_target = (self.target_position - position) / self.lattice_size
        
        # Distance to target (normalized)
        distance_to_target = np.linalg.norm(relative_to_target)
        
        # Create observation vector: [norm_x, norm_y, rel_target_x, rel_target_y, distance]
        observation = np.array([
            relative_to_target[0],
            relative_to_target[1]
        ])

        #print(observation)
        
        return observation
    
    def compute_observable(self, colloids: List[Colloid]) -> List:
        """
        Compute observables for all relevant agents.
        
        This is the method that SwarmRL expects to be implemented.
        
        Parameters
        ----------
        colloids : List[Colloid]
            List of all agents
            
        Returns
        -------
        List[np.ndarray]
            List of observation arrays, one for each relevant agent
        """
        # Get indices of relevant agents
        reference_ids = self.get_colloid_indices(colloids)
        
        # Compute observations
        observables = [
            self.compute_single_observable(index, colloids) 
            for index in reference_ids
        ]
        
        #print(f"DEBUG: Observable computed for {len(observables)} agents")
        return observables
        
    def __call__(self, colloids: List[Colloid]) -> np.ndarray:
        """
        Compute observables and return as numpy array (backward compatibility).
        
        Parameters
        ----------
        colloids : List[Colloid]
            List of all agents
            
        Returns
        -------
        np.ndarray
            Observation matrix [n_agents, observation_dim]
        """
        observables = self.compute_observable(colloids)
        return np.array(observables)


class LatticeActorCriticNet(nn.Module):
    """
    Neural network for discrete lattice with 4 actions.
    """
    
    @nn.compact
    def __call__(self, x):
        # Shared layers
        x = nn.Dense(features=32)(x)
        x = nn.relu(x)
        x = nn.Dense(features=32)(x)
        x = nn.relu(x)
        
        # Actor head (4 actions: right, up, left, down)
        actor = nn.Dense(features=16)(x)
        actor = nn.relu(actor)
        actor = nn.Dense(features=4)(actor)
        
        # Critic head (value function)
        critic = nn.Dense(features=16)(x)
        critic = nn.relu(critic)
        critic = nn.Dense(features=1)(critic)
        
        return actor, critic


class LatticeForceFunction(srl.force_functions.ForceFunction):
    """
    Custom force function for discrete lattice that works with SwarmRL agents.
    
    This class inherits from ForceFunction and uses the standard calc_action method
    to get actions from SwarmRL agents, which return DirectionAction objects.
    """
    
    def __init__(self, agents):
        """
        Initialize the lattice force function.
        
        Parameters
        ----------
        agents : dict
            Dictionary of SwarmRL agents
        """
        super().__init__(agents)
        
    # Note: We inherit calc_action from the parent ForceFunction class
    # The discrete lattice engine will call this method and get back Action objects


def get_system_runner(system=None, cycle_index=0):
    """
    Create a discrete lattice engine for training.
    
    Parameters
    ----------
    system : None
        Unused, kept for compatibility
    cycle_index : int
        Cycle index for episodic training
        
    Returns
    -------
    DiscreteLatticeEngine
        Configured lattice engine
    """
    # Create engine
    engine = DiscreteLatticeEngine(
        lattice_size=sim_params['LATTICE_SIZE'],
        periodic=False,  # Use walls
        seed=sim_params['SEED'] + cycle_index,
    )
    
    # Add agents at random positions
    engine.add_agents_random(sim_params['N_AGENTS'], agent_type=0)
    
    return engine


def main():
    """Main training function."""
    print("=== Discrete Lattice RL Training ===")
    print(f"Lattice size: {sim_params['LATTICE_SIZE']}x{sim_params['LATTICE_SIZE']}")
    print(f"Number of agents: {sim_params['N_AGENTS']}")
    print(f"Episodes: {sim_params['N_EPISODES']}")
    
    # Create output directory
    out_dir = f"./lattice_training_{sim_params['FILENAME']}"
    pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)
    
    # Setup live visualization - declare variables at function level
    visualization_enabled = False
    plt = None
    fig = None
    ax1 = None
    ax2 = None
    
    try:
        import matplotlib.pyplot as plt
        plt.ion()  # Turn on interactive mode
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Left plot: Agent positions and target
        ax1.set_xlim(0, sim_params['LATTICE_SIZE'])
        ax1.set_ylim(0, sim_params['LATTICE_SIZE'])
        ax1.set_xlabel('X Position')
        ax1.set_ylabel('Y Position') 
        ax1.set_title('Agent Positions and Target')
        ax1.grid(True, alpha=0.3)
        ax1.set_aspect('equal')
        
        # Right plot: Rewards over time
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Average Reward')
        ax2.set_title('Training Progress')
        ax2.grid(True, alpha=0.3)
        
        visualization_enabled = True
        print("Live visualization enabled!")
    except ImportError:
        plt = None
        print("Matplotlib not available - running without visualization")
    
    # Create task and observable
    task = LatticeTargetSearchTask(
        lattice_size=sim_params['LATTICE_SIZE'],
        target_radius=sim_params['TARGET_RADIUS'],
        reward_scale_factor=10.0,  # Increased for stronger learning signals
        particle_type=0,
        sim_params=sim_params,
    )
    
    observable = LatticePositionObservable(
        lattice_size=sim_params['LATTICE_SIZE'],
        particle_type=0,
    )
    
    # Link observable to task's target
    def update_observable_target():
        observable.set_target_position(task.target_position)
    
    # Create neural network
    network = srl.networks.FlaxModel(
        flax_model=LatticeActorCriticNet(),
        optimizer=optax.adam(learning_rate=0.0001),
        input_shape=(1, 2),  # (batch_size, feature_dim) - 5-dimensional observation
    )
    
    # Define custom actions that carry direction information
    class DirectionAction(Action):
        """Custom action that stores direction index for discrete lattice."""
        def __init__(self, direction_index):
            super().__init__()
            self.direction_index = direction_index
    
    # Define actions with direction indices
    move_right = DirectionAction(0)  # Direction index 0 = right
    move_up = DirectionAction(1)     # Direction index 1 = up  
    move_left = DirectionAction(2)   # Direction index 2 = left
    move_down = DirectionAction(3)   # Direction index 3 = down
    
    actions = {
        "MoveRight": move_right,
        "MoveUp": move_up,
        "MoveLeft": move_left,
        "MoveDown": move_down,
    }
    
    # Create loss function
    loss = srl.losses.ProximalPolicyLoss(
        entropy_coefficient=0.02,
        epsilon=0.2
    )
    
    # Create agent
    agent = srl.agents.ActorCriticAgent(
        particle_type=0,
        network=network,
        task=task,
        observable=observable,
        actions=actions,
        loss=loss,
        train=True
    )
    
    # Create trainer
    rl_trainer = srl.trainers.EpisodicTrainer([agent])
    
    # Override the trainer's initialize_training method to use our custom force function
    original_init = rl_trainer.initialize_training
    def custom_initialize_training():
        return LatticeForceFunction(rl_trainer.agents)
    rl_trainer.initialize_training = custom_initialize_training
    
    # Override the update_rl method to use our custom force function
    original_update = rl_trainer.update_rl
    def custom_update_rl():
        reward = 0.0
        switches = []
        
        for agent in rl_trainer.agents.values():
            if isinstance(agent, srl.agents.ActorCriticAgent):
                ag_reward, ag_killed = agent.update_agent()
                reward += np.mean(ag_reward)
                switches.append(ag_killed)
        
        # Create new force function
        interaction_model = LatticeForceFunction(rl_trainer.agents)
        return interaction_model, np.array(reward), any(switches)
    
    rl_trainer.update_rl = custom_update_rl
    
    # Custom training loop to handle lattice-specific logic
    def custom_get_engine(system=None, cycle_index=0):
        engine = get_system_runner(system, cycle_index)
        
        # Reset target every reset_freq episodes
        if cycle_index % sim_params['RESET_FREQ'] == 0:
            task.reset_target()
            update_observable_target()
        
        return engine
    
    # Custom training loop with live visualization
    print("Starting training...")
    rewards = []
    
    # Create engine once and reuse it
    engine = get_system_runner(system=None, cycle_index=0)
    current_engine = engine
    
    # Set initial target
    task.reset_target()
    update_observable_target()
    print(f"Initial target: {task.target_position}")
    
    for episode in range(sim_params['N_EPISODES']):
        # Reset engine only every reset_freq episodes
        if episode % sim_params['RESET_FREQ'] == 0 and episode > 0:
            print(f"Resetting engine at episode {episode + 1}")
            # Reset target
            task.reset_target()
            update_observable_target()
            # Reset agent positions to random locations
            engine.reset_agents_to_random_positions()
            print(f"New target: {task.target_position}")
        
        # Run one episode - exactly EPISODE_LENGTH decisions and moves
        episode_rewards = []
        
        # Get the force function once per episode
        force_function = rl_trainer.initialize_training()
                
        # Run exactly EPISODE_LENGTH integration steps (decisions + moves)
        engine.integrate(n_slices=sim_params['EPISODE_LENGTH'], force_model=force_function)
        
        # Collect final rewards after the episode
        colloids = engine.colloids
        if colloids:
            final_rewards = task(colloids)
            episode_rewards.extend(final_rewards if isinstance(final_rewards, list) else [final_rewards])
        
        # Update RL agent after episode
        force_function, episode_reward, killed = rl_trainer.update_rl()
        avg_reward = np.mean(episode_rewards) if episode_rewards else episode_reward
        rewards.append(avg_reward)
        
        # Update visualization every episode
        if visualization_enabled and episode % 1 == 0:  # Update every episode
            try:
                # Get current positions
                colloids = engine.colloids
                if colloids:
                    positions = np.array([colloid.pos[:2] for colloid in colloids])  # Get x, y positions
                    
                    # Clear and update left plot (positions)
                    ax1.clear()
                    ax1.set_xlim(0, sim_params['LATTICE_SIZE'])
                    ax1.set_ylim(0, sim_params['LATTICE_SIZE'])
                    ax1.set_xlabel('X Position')
                    ax1.set_ylabel('Y Position')
                    ax1.set_title(f'Episode {episode + 1} - Agents and Target')
                    ax1.grid(True, alpha=0.3)
                    ax1.set_aspect('equal')
                    
                    # Plot agents
                    ax1.scatter(positions[:, 0], positions[:, 1], 
                              c='blue', s=100, alpha=0.7, label='Agents', marker='o')
                    
                    # Plot target
                    target_pos = task.target_position
                    ax1.scatter(target_pos[0], target_pos[1], 
                              c='red', s=200, alpha=0.8, label='Target', marker='*')
                    
                    # Add target radius circle
                    circle = plt.Circle(target_pos, sim_params['TARGET_RADIUS'], 
                                      fill=False, color='red', alpha=0.3, linestyle='--')
                    ax1.add_patch(circle)
                    
                    ax1.legend()
                    
                    # Update right plot (rewards)
                    ax2.clear()
                    ax2.plot(rewards, 'b-', alpha=0.7)
                    if len(rewards) > 10:
                        # Add running average
                        running_avg = np.convolve(rewards, np.ones(10)/10, mode='valid')
                        ax2.plot(range(9, len(rewards)), running_avg, 'r-', alpha=0.8, label='Running Avg (10)')
                        ax2.legend()
                    ax2.set_xlabel('Episode')
                    ax2.set_ylabel('Average Reward')
                    ax2.set_title(f'Training Progress - Reward: {avg_reward:.2f}')
                    ax2.grid(True, alpha=0.3)
                    
                    # Update display
                    plt.tight_layout()
                    plt.pause(0.01)  # Small pause to allow display update
                    
            except Exception as e:
                print(f"Visualization error: {e}")
        
        # Print progress
        if episode % 10 == 0 or episode == sim_params['N_EPISODES'] - 1:
            print(f"Episode {episode + 1}/{sim_params['N_EPISODES']}, "
                  f"Avg Reward: {avg_reward:.3f}, "
                  f"Target: [{target_pos[0]:.0f}, {target_pos[1]:.0f}]")
    
    # Keep final plot open
    if visualization_enabled:
        plt.ioff()  # Turn off interactive mode
        plt.show(block=False)  # Show final plot
    
    rewards = np.array(rewards)
    
    # Save results
    print("Saving results...")
    rl_trainer.export_models(out_dir)
    np.save(f"{out_dir}/rewards.npy", rewards)
    
    with open(f"{out_dir}/sim_params.json", 'w') as f:
        json.dump(sim_params, f, indent=2)
    
    # Print summary
    print(f"\nTraining completed!")
    print(f"Final average reward: {np.mean(rewards[-100:]):.2f}")
    print(f"Results saved to: {out_dir}")
    
    return rewards, (plt, fig, ax1, ax2) if visualization_enabled else (None, None, None, None)


if __name__ == "__main__":
    import time
    
    start_time = time.time()
    rewards, (plt, fig, ax1, ax2) = main()
    end_time = time.time()
    
    print(f"\nTotal training time: {end_time - start_time:.2f} seconds")
    
    # Save final plots
    if plt is not None:
        try:
            # Save the live visualization plots
            if fig is not None:
                fig.savefig(f"./lattice_training_{sim_params['FILENAME']}/live_training_visualization.png", dpi=150, bbox_inches='tight')
                print("Live training visualization saved!")
            
            # Also create a separate rewards plot
            plt.figure(figsize=(10, 6))
            plt.plot(rewards, 'b-', alpha=0.7, label='Episode Rewards')
            if len(rewards) > 10:
                running_avg = np.convolve(rewards, np.ones(10)/10, mode='valid')
                plt.plot(range(9, len(rewards)), running_avg, 'r-', alpha=0.8, linewidth=2, label='Running Average (10)')
            plt.title('Training Rewards Over Episodes')
            plt.xlabel('Episode')
            plt.ylabel('Average Reward')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.savefig(f"./lattice_training_{sim_params['FILENAME']}/training_rewards.png", dpi=150, bbox_inches='tight')
            print("Final reward plot saved!")
            
            # Keep plots open for inspection
            print("\nTraining completed! Close the plot windows to finish.")
            plt.show()  # This will block until windows are closed
            
        except Exception as e:
            print(f"Error saving plots: {e}")
    else:
        print("Matplotlib not available for plotting")
