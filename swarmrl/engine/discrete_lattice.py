"""
Module for a simple 2D discrete lattice engine.

This module provides a simple implementation of a discrete lattice-based
simulation engine where agents move on a 2D grid. This serves as a concrete
example of how to implement the TemplateEngine interface.
"""

import logging
import pathlib
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from swarmrl.force_functions import ForceFunction
from swarmrl.components import Colloid

from .template_engine import TemplateEngine

logger = logging.getLogger(__name__)


class LatticeAgent:
    """
    Simple agent that exists on a discrete lattice.
    
    Attributes
    ----------
    agent_id : int
        Unique identifier for the agent
    position : np.ndarray
        Current lattice position [x, y]
    velocity : np.ndarray
        Current velocity direction [vx, vy] (should be unit vector in cardinal direction)
    agent_type : int
        Type identifier for the agent
    """
    
    def __init__(self, agent_id: int, position: np.ndarray, agent_type: int = 0):
        self.agent_id = agent_id
        self.position = np.array(position, dtype=int)
        self.velocity = np.array([0, 0], dtype=int)  # Start stationary
        self.agent_type = agent_type


class DiscreteLatticeEngine(TemplateEngine):
    """
    A simple 2D discrete lattice simulation engine.
    
    This engine simulates agents moving on a 2D square lattice. Agents can move
    in 4 cardinal directions (up, down, left, right) and the force model sets
    the velocity direction. During each integration step, agents move to the
    next lattice node in their velocity direction.
    
    Parameters
    ----------
    lattice_size : int
        Size of the square lattice (lattice_size x lattice_size)
    periodic : bool, optional
        Whether to use periodic boundary conditions (default: True)
    """
    
    def __init__(
        self,
        lattice_size: int,
        periodic: bool = True,
        out_folder: Union[str, pathlib.Path] = ".",
        write_chunk_size: int = 100,
        seed: int = 42,
        **kwargs
    ):
        """
        Initialize the discrete lattice engine.
        
        Parameters
        ----------
        lattice_size : int
            Size of the square lattice
        periodic : bool
            Use periodic boundary conditions
        out_folder : Union[str, pathlib.Path]
            Output folder for data
        write_chunk_size : int
            Chunk size for writing data
        seed : int
            Random seed
        **kwargs
            Additional parameters
        """
        # Create a simple test_info dict to satisfy parent constructor
        test_info = {
            'lattice_size': lattice_size,
            'periodic': periodic,
            'seed': seed
        }
        
        # Create a simple params object with the seed attribute
        class SimpleParams:
            def __init__(self, seed):
                self.seed = seed
        
        self.params = SimpleParams(seed)
        
        super().__init__(test_info, out_folder, write_chunk_size, **kwargs)
        
        self.lattice_size = lattice_size
        self.periodic = periodic
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        
        # Initialize lattice and agents
        self.lattice = np.zeros((lattice_size, lattice_size), dtype=int)  # 0 = empty, agent_id = occupied
        self.agents = []  # List of LatticeAgent objects
        self.colloids = []  # List of Colloid objects for SwarmRL compatibility
        self.next_agent_id = 1
        
        # Cardinal directions: [right, up, left, down]
        self.directions = np.array([
            [1, 0],   # right
            [0, 1],   # up
            [-1, 0],  # left
            [0, -1]   # down
        ])
        
        # Simulation state
        self.current_time = 0.0
        self.integration_initialized = False
        
        logger.info(f"Initialized DiscreteLatticeEngine with {lattice_size}x{lattice_size} lattice")
    
    def _init_simulation(self):
        """Initialize the simulation system."""
        self.integration_initialized = True
        logger.info("Simulation initialized")
    
    def _check_initialization(self):
        """Check if simulation is already initialized."""
        if self.integration_initialized:
            raise RuntimeError("Cannot modify engine after integration has started")
    
    def _is_valid_position(self, position: np.ndarray) -> bool:
        """Check if a position is valid on the lattice."""
        x, y = position
        return 0 <= x < self.lattice_size and 0 <= y < self.lattice_size
    
    def _wrap_position(self, position: np.ndarray) -> np.ndarray:
        """Wrap position for periodic boundary conditions."""
        if self.periodic:
            return np.array([
                position[0] % self.lattice_size,
                position[1] % self.lattice_size
            ])
        return position
    
    def _is_position_occupied(self, position: np.ndarray) -> bool:
        """Check if a lattice position is occupied by an agent."""
        x, y = position
        if not self._is_valid_position(position):
            return True  # Consider out-of-bounds as occupied
        return self.lattice[x, y] != 0
    
    def add_particle_at_position(
        self,
        position: Union[List[int], np.ndarray],
        particle_type: int = 0,
        **kwargs
    ) -> int:
        """
        Add a single agent at a specific lattice position.
        
        Parameters
        ----------
        position : Union[List[int], np.ndarray]
            Lattice coordinates [x, y]
        particle_type : int
            Type identifier for the agent
        **kwargs
            Additional agent properties (ignored)
        
        Returns
        -------
        int
            Unique identifier for the added agent
        """
        self._check_initialization()
        
        position = np.array(position, dtype=int)
        
        # Check if position is valid and not occupied
        if not self._is_valid_position(position):
            raise ValueError(f"Position {position} is outside lattice bounds")
        
        if self._is_position_occupied(position):
            raise ValueError(f"Position {position} is already occupied")
        
        # Create new agent
        agent_id = self.next_agent_id
        self.next_agent_id += 1
        
        agent = LatticeAgent(agent_id, position, particle_type)
        self.agents.append(agent)
        
        # Create corresponding Colloid object for SwarmRL compatibility
        colloid = Colloid(
            pos=np.array([position[0], position[1], 0], dtype=float),  # Add z=0 for 3D compatibility
            director=np.array([1, 0, 0], dtype=float),  # Default director
            id=agent_id,
            velocity=np.array([0, 0, 0], dtype=float),  # Default velocity
            type=particle_type
        )
        self.colloids.append(colloid)
        
        # Update lattice
        x, y = position
        self.lattice[x, y] = agent_id
        
        logger.debug(f"Added agent {agent_id} at position {position}")
        return agent_id
    
    def add_agents_random(
        self,
        n_agents: int,
        agent_type: int = 0,
        **kwargs
    ) -> List[int]:
        """
        Add multiple agents at random positions.
        
        Parameters
        ----------
        n_agents : int
            Number of agents to add
        agent_type : int
            Type identifier for agents
        **kwargs
            Additional parameters (ignored)
        
        Returns
        -------
        List[int]
            List of agent IDs
        """
        self._check_initialization()
        
        agent_ids = []
        attempts = 0
        max_attempts = self.lattice_size * self.lattice_size * 10
        
        for i in range(n_agents):
            placed = False
            while not placed and attempts < max_attempts:
                x = self.rng.integers(0, self.lattice_size)
                y = self.rng.integers(0, self.lattice_size)
                position = np.array([x, y])
                
                if not self._is_position_occupied(position):
                    agent_id = self.add_particle_at_position(position, agent_type)
                    agent_ids.append(agent_id)
                    placed = True
                
                attempts += 1
            
            if not placed:
                logger.warning(f"Could not place agent {i+1}/{n_agents} after {max_attempts} attempts")
                break
        
        logger.info(f"Added {len(agent_ids)} agents randomly on lattice")
        return agent_ids
    
    def manage_forces(self, force_model: Optional[ForceFunction] = None) -> bool:
        """
        Manage external forces using SwarmRL force model.
        
        This method follows the same pattern as EspressoMD: it calls force_model.calc_action()
        with colloid objects and gets back Action objects.
        
        Parameters
        ----------
        force_model : Optional[ForceFunction]
            SwarmRL force function that computes actions for colloids
        
        Returns
        -------
        bool
            True if forces were successfully updated
        """
        if force_model is None or len(self.colloids) == 0:
            return False
        
        try:
            # Debug: Check the force model and agents
            logger.info(f"Force model type: {type(force_model)}")
            logger.info(f"Force model agents: {force_model.agents}")
            logger.info(f"Number of colloids: {len(self.colloids)}")
            
            # Call the force model to get actions for all colloids
            actions = force_model.calc_action(self.colloids)
            
            # Apply actions to agents
            for i, (action, agent) in enumerate(zip(actions, self.agents)):
                if hasattr(action, 'direction_index'):
                    # Use custom direction index if available
                    direction_idx = action.direction_index
                    if 0 <= direction_idx <= 3:
                        agent.velocity = self.directions[direction_idx].copy()
                    else:
                        agent.velocity = np.array([0, 0])  # Stay in place
                else:
                    # Fallback: use random direction
                    agent.velocity = self.directions[self.rng.integers(0, 4)].copy()
                    
            return True
            
        except Exception as e:
            logger.error(f"Error in manage_forces: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return False
    
    def integrate(
        self,
        n_slices: int,
        force_model: Optional[ForceFunction] = None
    ) -> None:
        """
        Integrate the simulation for n_slices time steps.
        
        Each integration step:
        1. Update forces/velocities using force model
        2. Move agents to next positions based on velocities
        3. Handle boundary conditions
        
        Parameters
        ----------
        n_slices : int
            Number of time steps to integrate
        force_model : Optional[ForceFunction]
            Force function to apply
        """
        if not self.integration_initialized:
            self._init_simulation()
        
        for step in range(n_slices):
            # Update velocities based on force model
            if force_model is not None:
                self.manage_forces(force_model)
            
            # Move agents
            self._move_agents()
            
            # Update time
            self.current_time += 1.0
        
        logger.debug(f"Integrated {n_slices} steps, current time: {self.current_time}")
    
    def _move_agents(self):
        """Move all agents based on their current velocities."""
        # Clear lattice
        self.lattice.fill(0)
        
        # Calculate new positions
        new_positions = []
        for agent in self.agents:
            new_pos = agent.position + agent.velocity
            
            # Handle boundary conditions
            if self.periodic:
                new_pos = self._wrap_position(new_pos)
            else:
                # Bounce off walls by staying in place
                if not self._is_valid_position(new_pos):
                    new_pos = agent.position.copy()
            
            new_positions.append(new_pos)
        
        # Handle collisions - if multiple agents try to move to same position,
        # randomly select one to move, others stay
        position_claims = {}
        for i, new_pos in enumerate(new_positions):
            pos_key = tuple(new_pos)
            if pos_key not in position_claims:
                position_claims[pos_key] = []
            position_claims[pos_key].append(i)
        
        # Resolve conflicts
        final_positions = []
        for i, agent in enumerate(self.agents):
            new_pos = new_positions[i]
            pos_key = tuple(new_pos)
            
            if len(position_claims[pos_key]) == 1:
                # No conflict, agent can move
                final_positions.append(new_pos)
            else:
                # Conflict - randomly choose one agent to move
                chosen_agent = self.rng.choice(position_claims[pos_key])
                if i == chosen_agent:
                    final_positions.append(new_pos)
                else:
                    final_positions.append(agent.position.copy())  # Stay in place
        
        # Update agent positions and lattice
        for i, agent in enumerate(self.agents):
            agent.position = final_positions[i]
            x, y = agent.position
            if self._is_valid_position(agent.position):
                self.lattice[x, y] = agent.agent_id
            
            # Update corresponding colloid
            if i < len(self.colloids):
                # Create new colloid with updated position (Colloid is frozen dataclass)
                self.colloids[i] = Colloid(
                    pos=np.array([agent.position[0], agent.position[1], 0], dtype=float),
                    director=np.array([agent.velocity[0], agent.velocity[1], 0], dtype=float) if np.any(agent.velocity) else np.array([1, 0, 0], dtype=float),
                    id=agent.agent_id,
                    velocity=np.array([agent.velocity[0], agent.velocity[1], 0], dtype=float),
                    type=agent.agent_type
                )
    
    def get_particle_data(self) -> Dict[str, np.ndarray]:
        """
        Get current agent data.
        
        Returns
        -------
        Dict[str, np.ndarray]
            Dictionary containing:
            - 'id': agent IDs
            - 'type': agent types  
            - 'position': positions [N, 2]
            - 'velocity': velocities [N, 2]
        """
        if len(self.agents) == 0:
            return {
                'id': np.array([], dtype=int),
                'type': np.array([], dtype=int),
                'position': np.zeros((0, 2), dtype=int),
                'velocity': np.zeros((0, 2), dtype=int)
            }
        
        return {
            'id': np.array([agent.agent_id for agent in self.agents]),
            'type': np.array([agent.agent_type for agent in self.agents]),
            'position': np.array([agent.position for agent in self.agents]),
            'velocity': np.array([agent.velocity for agent in self.agents])
        }
    
    def get_system_state(self) -> Dict[str, Any]:
        """
        Get complete system state.
        
        Returns
        -------
        Dict[str, Any]
            System state information
        """
        return {
            'lattice_size': self.lattice_size,
            'periodic': self.periodic,
            'current_time': self.current_time,
            'n_agents': len(self.agents),
            'lattice_occupancy': np.sum(self.lattice > 0) / (self.lattice_size ** 2),
            'lattice_state': self.lattice.copy()
        }
    
    def reset_simulation(self):
        """Reset simulation to initial state."""
        self.lattice.fill(0)
        self.agents.clear()
        self.colloids.clear()
        self.next_agent_id = 1
        self.current_time = 0.0
        self.integration_initialized = False
        logger.info("Simulation reset")
    
    def finalize(self):
        """Clean up and finalize simulation."""
        logger.info(f"Finalizing simulation with {len(self.agents)} agents at time {self.current_time}")
        # Could save final state, write output files, etc.
        pass
    
    def visualize_lattice(self) -> np.ndarray:
        """
        Get a visualization of the current lattice state.
        
        Returns
        -------
        np.ndarray
            2D array showing agent positions (0 = empty, agent_id = occupied)
        """
        return self.lattice.copy()
    
    def get_agent_by_id(self, agent_id: int) -> Optional[LatticeAgent]:
        """Get agent by ID."""
        for agent in self.agents:
            if agent.agent_id == agent_id:
                return agent
        return None
    
    def reset_agents_to_random_positions(self):
        """Reset all agents to new random positions."""
        if not self.agents:
            return
        
        # Clear lattice
        self.lattice.fill(0)
        
        # Reset agents to new random positions
        for i, agent in enumerate(self.agents):
            # Find a random unoccupied position
            while True:
                new_pos = self.rng.integers(0, self.lattice_size, size=2)
                if not self._is_position_occupied(new_pos):
                    break
            
            # Update agent position and reset velocity
            agent.position = new_pos.copy()
            agent.velocity = np.array([0, 0], dtype=int)
            
            # Update lattice
            self.lattice[new_pos[0], new_pos[1]] = agent.agent_id
            
            # Update corresponding colloid
            if i < len(self.colloids):
                self.colloids[i] = Colloid(
                    pos=np.array([new_pos[0], new_pos[1], 0], dtype=float),
                    director=np.array([1, 0, 0], dtype=float),
                    id=agent.agent_id,
                    type=agent.agent_type
                )
        
        logger.info(f"Reset {len(self.agents)} agents to random positions")
