"""
Module for a template simulation engine.

This module provides a generic template for simulation engines that can be used
in reinforcement learning applications. It follows the same structure as the
EspressoMD engine but without specific implementation details.
"""

import dataclasses
import logging
import pathlib
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from swarmrl.force_functions import ForceFunction

from .engine import Engine

logger = logging.getLogger(__name__)


class TemplateEngine(Engine):
    """
    A template simulation engine.
    
    This class provides the structure and interface for simulation engines
    that can be used in reinforcement learning applications. It inherits
    from the base Engine class and defines all necessary methods with
    proper signatures but no specific implementation.
    
    The class is designed to be subclassed for specific simulation types
    (molecular dynamics, agent-based models, Monte Carlo, etc.).
    """
    
    def __init__(
        self,
        test_info: dict,
        out_folder: Union[str, pathlib.Path] = ".",
        write_chunk_size: int = 100,
        **kwargs
    ):
        """
        Constructor for the template engine.
        
        Parameters
        ----------
        params : GeneralParams
            Parameter object containing simulation configuration
        out_folder : Union[str, pathlib.Path]
            Path to output folder for data storage
        write_chunk_size : int
            Chunk size for data writing operations
        **kwargs
            Additional keyword arguments for specific implementations
        """
        self.out_folder = pathlib.Path(out_folder).resolve()
        self.write_chunk_size = write_chunk_size
        
        # Initialize random number generator
        self.rng = np.random.default_rng(self.params.seed)
        
        # Initialize containers for simulation objects
        self.agents = []

        
        # Simulation state
        self.current_time = 0.0
        self.integration_initialized = False
        
        # Store additional parameters
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    def _init_simulation(self):
        """
        Initialize the simulation system.
        
        This method should set up the basic simulation parameters,
        boundary conditions, and any necessary data structures.
        """
        raise NotImplementedError("Subclasses must implement _init_simulation")
    
    def _check_initialization(self):
        """
        Check if the simulation has been initialized.
        
        Raises
        ------
        RuntimeError
            If simulation is already initialized and modifications are attempted
        """
        if self.integration_initialized:
            raise RuntimeError(
                "Cannot modify engine after integration has started"
            )
    
    # Particle/Agent Management Methods
    
    def add_particle_at_position(
        self,
        position: Union[List[float], np.ndarray],
        particle_type: int = 0,
        radius: float = 1.0,
        mass: float = 1.0,
        **kwargs
    ):
        """
        Add a single particle/agent at a specific position.
        
        Parameters
        ----------
        position : Union[List[float], np.ndarray]
            Initial position of the particle
        particle_type : int
            Type identifier for the particle
        radius : float
            Particle radius
        mass : float
            Particle mass
        **kwargs
            Additional particle properties
        
        Returns
        -------
        particle_id : int
            Unique identifier for the added particle
        """
        raise NotImplementedError("Subclasses must implement add_particle_at_position")
    
    # Core Simulation Methods
    
    def manage_forces(self, force_model: Optional[ForceFunction] = None) -> bool:
        """
        Update forces from the force model.
        
        Parameters
        ----------
        force_model : Optional[ForceFunction]
            Force function to apply to particles
        
        Returns
        -------
        bool
            True if forces were successfully updated
        """
        raise NotImplementedError("Subclasses must implement manage_forces")
    
    def integrate(
        self,
        n_slices: int,
        force_model: Optional[ForceFunction] = None
    ) -> None:
        """
        Integrate the simulation for a given number of time slices.
        
        Parameters
        ----------
        n_slices : int
            Number of time slices to integrate
        force_model : Optional[ForceFunction]
            Force function to apply during integration
        """
        raise NotImplementedError("Subclasses must implement integrate")
    
    def get_particle_data(self) -> Dict[str, np.ndarray]:
        """
        Get current particle data.
        
        Returns
        -------
        Dict[str, np.ndarray]
            Dictionary containing particle properties:
            - 'id': particle IDs
            - 'type': particle types
            - 'position': positions
            - 'velocity': velocities (if applicable)
            - 'director': orientations (if applicable)
        """
        raise NotImplementedError("Subclasses must implement get_particle_data")
    
    def get_system_state(self) -> Dict[str, Any]:
        """
        Get complete system state information.
        
        Returns
        -------
        Dict[str, Any]
            Dictionary containing system state information
        """
        raise NotImplementedError("Subclasses must implement get_system_state")
        
    def finalize(self):
        """
        Clean up and finalize the simulation.
        
        This method should handle any necessary cleanup operations
        like writing final data, closing files, etc.
        """
        raise NotImplementedError("Subclasses must implement finalize")
    
