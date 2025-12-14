import dataclasses
import pathlib
import shelve
import typing
import logging
import time

import h5py
import numba
import numpy as np
import pint
import scipy.integrate
import jax
import tqdm

from .engine import Engine
from swarmrl.actions import MPIAction
from swarmrl.components.colloid import Colloid
from swarmrl.force_functions.global_force_fn import GlobalForceFunction

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class GauravSimParams:
    """
    time_step < snapshot_interval < time_slice, all multiples of time_step.

    capillary_force_data_path: path to the database that contains precalculated
        force and torque data. Specify path **without** the file extension.
        The shelve module will find all files needed to load the database.
    """

    ureg: pint.UnitRegistry
    box_length: float
    time_step: float
    time_slice: float
    snapshot_interval: float
    raft_radius: float
    raft_repulsion_strength: float
    wall_repulsion_strength: float
    dynamic_viscosity: float
    fluid_density: float
    lubrication_threshold: float
    magnetic_constant: float
    capillary_force_data_path: pathlib.Path


def setup_unit_system(ureg: pint.UnitRegistry):
    """
    In-place definition of the unit system for the simulation,
    added to the ureg object.
    """
    # basis units
    ureg.define("sim_length = 1 micrometer")
    ureg.define("sim_time = 1 second")
    ureg.define("sim_current = 1 ampere")
    ureg.define("sim_force = newton")

    # gaurav's original unit system isn't consistent, but this should
    # only affect the water density

    # derived units
    ureg.define("sim_mass = sim_force * sim_time**2 / sim_length")
    ureg.define("sim_velocity = sim_length / sim_time")
    ureg.define("sim_angular_velocity = 1 / sim_time")
    ureg.define("sim_dyn_viscosity = sim_mass / (sim_length * sim_time)")
    ureg.define("sim_kin_viscosity = sim_length**2 / sim_time")
    ureg.define("sim_torque = sim_length * sim_force")
    ureg.define("sim_magnetic_field = sim_mass / sim_time**2 / sim_current")
    ureg.define("sim_magnetic_permeability = sim_force / sim_current **2")
    ureg.define("sim_magnetic_moment = sim_current * sim_length**2")

    return ureg


def convert_params_to_simunits(params: GauravSimParams) -> GauravSimParams:
    ureg = params.ureg
    params_simunits = GauravSimParams(
        ureg=ureg,
        box_length=params.box_length.m_as("sim_length"),
        time_step=params.time_step.m_as("sim_time"),
        time_slice=params.time_slice.m_as("sim_time"),
        snapshot_interval=params.snapshot_interval.m_as("sim_time"),
        raft_radius=params.raft_radius.m_as("sim_length"),
        raft_repulsion_strength=params.raft_repulsion_strength.m_as("sim_force"),
        wall_repulsion_strength=params.wall_repulsion_strength.m_as("sim_force"),
        dynamic_viscosity=params.dynamic_viscosity.m_as("sim_dyn_viscosity"),
        fluid_density=params.fluid_density.m_as("sim_mass / sim_length**3"),
        lubrication_threshold=params.lubrication_threshold.m_as("sim_length"),
        magnetic_constant=params.magnetic_constant.m_as("sim_magnetic_permeability"),
        capillary_force_data_path=params.capillary_force_data_path,
    )
    return params_simunits


@dataclasses.dataclass
class Raft:
    pos: np.array
    alpha: float  # in radians
    magnetic_moment: float
    rotational_velocity: float = 0.0

    def get_director(self) -> np.array:
        return np.array([np.cos(self.alpha), np.sin(self.alpha)])


class Model:
    def calc_action(self, rafts: typing.List[Raft]) -> MPIAction:
        raise NotImplementedError


class ConstantAction(Model):
    def __init__(self, action: MPIAction) -> None:
        self.action = action

    def calc_action(self, rafts: typing.List[Raft]) -> MPIAction:
        return self.action


def calc_B_field(action: MPIAction, time: float):
    
    #TODO replace MPI Action?
    frequency = action.frequency
    phase = action.phase
    mag_f = action.magnitude
    offset = action.offset
    mag = [mag_f[0] * np.cos(frequency[0] * 2*np.pi * time - phase[0]) + offset[0],
           mag_f[1] * np.cos(frequency[1] * 2*np.pi * time - phase[1]) + offset[1]]

    return mag



class GauravSim(Engine):
    """
    Based on Gaurav's implementation of raft motion described in
    "Order and information in the patterns of spinning magnetic
    micro-disks at the air-water interface"
    doi:10.1126/sciadv.abk0685
    """

    def __init__(
        self,
        params: GauravSimParams,
        out_folder: typing.Union[str, pathlib.Path],
        h5_group_tag: str = "rafts",
        with_precalc_capillary: bool = True,
        save_h5: bool = True,
        write_chunk_size: int = 1000,
    ):
        setup_unit_system(params.ureg)
        self.params: GauravSimParams = convert_params_to_simunits(params)
        self.colloids: typing.List[Colloid] = []

        self.out_folder: pathlib.Path = pathlib.Path(out_folder).resolve()
        self.h5_group_tag = h5_group_tag
        self.with_precalc_capillary = with_precalc_capillary

        self.integration_initialised = False
        self.slice_idx = None
        self.current_action: MPIAction = None
        
        self.save_h5 = save_h5
        self.write_chunk_size = write_chunk_size

    def add_colloids(self, pos: np.array, alpha: float, magnetic_moment: float):
        self._check_already_initialised()
        self.colloids.append(
            Raft(
                pos=pos.m_as("sim_length"),
                alpha=alpha,
                magnetic_moment=magnetic_moment.m_as("sim_magnetic_moment"),
                rotational_velocity=0 #rotational_velocity.m_as("sim_angular_velocity")
            )
        )

    def _get_state_from_rafts(self):
        return np.array([[r.pos[0], r.pos[1], r.alpha] for r in self.colloids])

    def _update_rafts_from_state(self, state):
        for r, s in zip(self.colloids, state):
            r.pos = s[:2]
            r.rotational_velocity = (s[2] - r.alpha) / self.params.time_slice
            r.alpha = s[2]


    def _check_already_initialised(self):
        if self.integration_initialised:
            raise RuntimeError(
                "You cannot change the system configuration "
                "after the first call to integrate()"
            )

    def _init_h5_output(self):
        """
        Initialize the hdf5 output.

        This method will create a directory for the data to be stored within. Follwing
        this, a hdf5 database is constructed for storing of the simulation data.
        """
        self.h5_filename = self.out_folder / "trajectory.hdf5"
        self.out_folder.mkdir(parents=True, exist_ok=True)
        if self.h5_filename.exists():
            self.h5_filename.unlink()

        n_rafts = len(self.colloids)

        self.buffer_size = self.write_chunk_size  # Write to disk every 1000 steps
        self.data_buffer = {
            "Times": [],
            "Alphas": [],
            "Angular_Velocity": [],
            "Unwrapped_Positions": [],
            "Magnetic_field": [],
            "Time_to_keep": []
        }

        # create datasets with 3 dimension regardless of data dimension to make
        # data handling easier later
        with h5py.File(self.h5_filename.as_posix(), "w") as h5_outfile:
            ids = np.arange(len(self.colloids))
            h5_outfile.create_dataset("Ids", data=ids)
            
            part_group = h5_outfile.require_group(self.h5_group_tag)
            ds_kwargs = dict(compression="lzf", chunks=True)
            n_rafts = len(self.colloids)

            part_group.create_dataset("Times", shape=(0, 1, 1), maxshape=(None, 1, 1), dtype=float, **ds_kwargs)
            part_group.create_dataset("Alphas", shape=(0, n_rafts, 1), maxshape=(None, n_rafts, 1), dtype=float, **ds_kwargs)
            part_group.create_dataset("Angular_Velocity", shape=(0, n_rafts, 1), maxshape=(None, n_rafts, 1), dtype=float, **ds_kwargs)
            part_group.create_dataset("Unwrapped_Positions", shape=(0, n_rafts, 2), maxshape=(None, n_rafts, 2), dtype=float, **ds_kwargs)

            act_group = h5_outfile.create_group("actions")
            act_group.create_dataset("Magnetic_field", shape=(0, 2), maxshape=(None, 2), dtype=float, **ds_kwargs)
            act_group.create_dataset("Time_to_keep", shape=(0, 1), maxshape=(None, 1), dtype=float, **ds_kwargs)

    def write_to_h5(
        self, traj_state_flat: np.array, times: np.array, action: MPIAction
    ):
        n_new_snapshots = len(times)
        num_particles = len(self.colloids)

        traj_reshaped = traj_state_flat.T.reshape((n_new_snapshots, num_particles, 3))

        self.data_buffer["Times"].append(times) 
        self.data_buffer["Alphas"].append(traj_reshaped[:, :, 2])
        self.data_buffer["Unwrapped_Positions"].append(traj_reshaped[:, :, :2])

        ang_vels = np.array([r.rotational_velocity for r in self.colloids])
        ang_vels_stacked = np.tile(ang_vels, (n_new_snapshots, 1))
        self.data_buffer["Angular_Velocity"].append(ang_vels_stacked)

        self.data_buffer["Magnetic_field"].append(np.tile(action.magnetic_field, (n_new_snapshots, 1)))
        self.data_buffer["Time_to_keep"].append(np.tile([action.keep_magnetic_field], (n_new_snapshots, 1)))
        
        if len(self.data_buffer["Times"]) * n_new_snapshots >= self.buffer_size:
            self._flush_buffer()

    def _flush_buffer(self):
        """

        """
        if not self.data_buffer["Times"]:
            return


        chunk_times = np.concatenate(self.data_buffer["Times"])
        chunk_alphas = np.concatenate(self.data_buffer["Alphas"])
        chunk_pos = np.concatenate(self.data_buffer["Unwrapped_Positions"])
        chunk_ang = np.concatenate(self.data_buffer["Angular_Velocity"])
        chunk_mag = np.concatenate(self.data_buffer["Magnetic_field"])
        chunk_keep = np.concatenate(self.data_buffer["Time_to_keep"])

        n_new = len(chunk_times)

        with h5py.File(self.h5_filename, "a") as f:
            pg = f[self.h5_group_tag]
            ag = f["actions"]

            # Helper to resize and write
            def append_ds(dataset, data):
                dataset.resize(dataset.shape[0] + n_new, axis=0)
                dataset[-n_new:] = data

            append_ds(pg["Times"], chunk_times[:, None, None])
            append_ds(pg["Alphas"], chunk_alphas[:, :, None])
            append_ds(pg["Angular_Velocity"], chunk_ang[:, :, None])
            append_ds(pg["Unwrapped_Positions"], chunk_pos)
            
            append_ds(ag["Magnetic_field"], chunk_mag)
            append_ds(ag["Time_to_keep"], chunk_keep)

        # Clear buffer
        for k in self.data_buffer:
            self.data_buffer[k] = []

    def finalize_h5(self):
        # Call this at the very end of simulation to save remaining steps
        self._flush_buffer()

    def _load_cap_forces(self):
        with shelve.open(self.params.capillary_force_data_path.as_posix()) as tempShelf:
            capillaryEEDistances = tempShelf["eeDistanceCombined"]  # unit: m
            capillaryForcesDistancesAsRowsLoaded = tempShelf[
                "forceCombinedDistancesAsRowsAll360"
            ]  # unit: N
            capillaryTorquesDistancesAsRowsLoaded = tempShelf[
                "torqueCombinedDistancesAsRowsAll360"
            ]  # unit: N.m

        capillaryEEDistances = np.insert(capillaryEEDistances, 0, 0)
        capillaryForcesDistancesAsRows = np.concatenate(
            (capillaryForcesDistancesAsRowsLoaded[:1, :], capillaryForcesDistancesAsRowsLoaded), axis=0)
        capillaryTorquesDistancesAsRows = np.concatenate(
            (capillaryTorquesDistancesAsRowsLoaded[:1, :], capillaryTorquesDistancesAsRowsLoaded), axis=0)

        # add angle=360, the same as angle = 0
        capillaryForcesDistancesAsRows = np.concatenate(
            (capillaryForcesDistancesAsRows, capillaryForcesDistancesAsRows[:, 0].reshape(1001, 1)), axis=1)
        capillaryTorquesDistancesAsRows = np.concatenate(
            (capillaryTorquesDistancesAsRows, capillaryTorquesDistancesAsRows[:, 0].reshape(1001, 1)), axis=1)

        # correct for the negative sign of the torque
        capillaryTorquesDistancesAsRows = - capillaryTorquesDistancesAsRows

        # some extra treatment for the force matrix
        # note the sharp transition at the peak-peak position (45 deg): only 1 deg difference,
        nearEdgeSmoothingThres = 1  # unit: micron; if 1, then it is equivalent to no smoothing.
        for distanceToEdge in np.arange(nearEdgeSmoothingThres):
            capillaryForcesDistancesAsRows[distanceToEdge, :] = capillaryForcesDistancesAsRows[nearEdgeSmoothingThres, :]
            capillaryTorquesDistancesAsRows[distanceToEdge, :] = capillaryTorquesDistancesAsRows[nearEdgeSmoothingThres, :]

        # select a cut-off distance below which all the attractive force (negative-valued) becomes zero,
        # due to raft wall-wall repulsion
        capAttractionZeroCutoff = 0
        mask = np.concatenate((capillaryForcesDistancesAsRows[:capAttractionZeroCutoff, :] < 0,
                            np.zeros((capillaryForcesDistancesAsRows.shape[0] - capAttractionZeroCutoff,
                                        capillaryForcesDistancesAsRows.shape[1]), dtype=int)),
                            axis=0)
        capillaryForcesDistancesAsRows[mask.nonzero()] = 0


        # realign the first peak-peak direction with an angle = capillaryPeakOffset from the x-axis.
        capillaryPeakOffset = 0
        capillaryForcesDistancesAsRows = np.roll(capillaryForcesDistancesAsRows, capillaryPeakOffset,
                                                axis=1)  # 45 is due to original data
        capillaryTorquesDistancesAsRows = np.roll(capillaryTorquesDistancesAsRows, capillaryPeakOffset, axis=1)

        capillaryEEDistances = self.params.ureg.Quantity(capillaryEEDistances, "meter")
        capillaryForcesDistancesAsRows = self.params.ureg.Quantity(capillaryForcesDistancesAsRows, "newton")
        capillaryTorquesDistancesAsRows = self.params.ureg.Quantity(capillaryTorquesDistancesAsRows, "newton * meter")
    
        self.capillary_distance = capillaryEEDistances.m_as("sim_length")
        self.capillary_force = capillaryForcesDistancesAsRows.m_as("sim_force")
        self.capillary_torque = capillaryTorquesDistancesAsRows.m_as("sim_torque")
    
    
    def RHS(self, t, state_flat, magFieldDir1, magFieldStrength1):
                
        return compute_derivatives_numba(
            t, state_flat, self.num_colloids, magFieldDir1, magFieldStrength1,
            # Constants
            self.params.raft_radius,
            self.params.lubrication_threshold,
            self.params.fluid_density,
            self.params.dynamic_viscosity,
            self.params.magnetic_constant,
            self.colloid_mag_moment,
            1.0, 1.0, 1.0, 1.0, 1.0, 1.0, # cm, cc, ch, tb, tm, tc
            self.mag_dp_torque, 
            self.mag_dp_force_on_axis, 
            self.mag_dp_force_off_axis,
            self.capillary_force,
            self.capillary_torque,
            self.lubA, self.lubB, self.lubC, self.lubG, self.lubCoeff,
            self.params.box_length,
            self.params.wall_repulsion_strength,
            0, # forceDueToCurvature
        )
    
    def _calc_mag(self):
        """
        Calculate the magnetic dipole forces/torque values. Be careful this only works
        for all magnetic moments being the same.
        """
        # Pick first colloid
        magmom = self.colloids[0].magnetic_moment
        mu0 = self.params.magnetic_constant

        # Dont ask, i dont know why. Done like this in the old code
        magDipoleEEDistances = self.params.ureg.Quantity(np.arange(0, 10001), "micrometer").m_as("sim_length")  
        magDipoleCCDistances = magDipoleEEDistances + self.params.raft_radius * 2

        orientationAngles = np.arange(0, 361)  # unit: degree;
        orientationAnglesInRad = np.radians(orientationAngles)

        magDpForceOnAxis = np.zeros((len(magDipoleEEDistances), len(orientationAngles)))
        magDpForceOffAxis = np.zeros((len(magDipoleEEDistances), len(orientationAngles)))
        magDpTorque = np.zeros((len(magDipoleEEDistances), len(orientationAngles)))

        for index, d in enumerate(magDipoleCCDistances):
            magDpForceOnAxis[index, :] = \
                3 * mu0 * magmom ** 2 * (1 - 3 * (np.cos(orientationAnglesInRad) ** 2)) / (4 * np.pi * d ** 4)
            magDpForceOffAxis[index, :] = \
                3 * mu0 * magmom ** 2 * (2 * np.cos(orientationAnglesInRad) *
                                                        np.sin(orientationAnglesInRad)) / (4 * np.pi * d ** 4)
            magDpTorque[index, :] = \
                mu0 * magmom ** 2 * (3 * np.cos(orientationAnglesInRad) *
                                                    np.sin(orientationAnglesInRad)) / (4 * np.pi * d ** 3)

        self.mag_dp_force_on_axis = magDpForceOnAxis
        self.mag_dp_force_off_axis = magDpForceOffAxis
        self.mag_dp_torque = magDpTorque

    def _calc_lub(self):
        stepSizeForDist = 0.1 # in micron, resolution parameter
        R = self.params.raft_radius

        lubCoeff = 1 / stepSizeForDist

        eeDistancesForCoeff = self.params.ureg.Quantity(np.arange(0, 15 + stepSizeForDist, stepSizeForDist, dtype='double'), "micrometer")  # unit: micron
        eeDistancesForCoeff[0] = self.params.ureg.Quantity(1e-10, "micrometer")
        eeDistancesForCoeff = eeDistancesForCoeff.m_as("sim_length")

        x = (eeDistancesForCoeff / R)  # unit: 1

        logx = np.log(x)

        lubA = x * (-0.285524 * x + 0.095493 * x * logx + 0.106103) / R  # unit: 1/um

        lubB = ((0.0212764 * (- logx) + 0.157378) * (- logx) + 0.269886) / (
                R * (- logx) * ((- logx) + 6.0425) + 6.32549
                )  # unit: 1/um

        lubG = ((0.0212758 * (- logx) + 0.181089) * (- logx) + 0.381213) / (
                R ** 3 * ((- logx) * ((- logx) + 6.0425) + 6.32549)
                )  # unit: 1/um^3

        lubC = - R * lubG

        self.lubA = lubA
        self.lubB = lubB
        self.lubC = lubC
        self.lubG = lubG
        self.lubCoeff = lubCoeff

    def integrate(self, n_slices: int, model: GlobalForceFunction):
        if not self.integration_initialised:
            self.time = 0.0
            self.slice_idx = 0

            # TODO be careful, because maybe num colloids changes?
            self.num_colloids = len(self.colloids)

            # TODO: check if timesteps and snapshots fit into timeslice
            self.timesteps_per_snapshot = int(
                round(self.params.snapshot_interval / self.params.time_step)
                )

            self.timesteps_per_slice = int(
                round(self.params.time_slice / self.params.time_step)
                )

            # TODO: maybe include a check that all magnetic moments are the same,
            # otherwise it causes some problems with the magnetic_dipole force.
            # If we want random magnetic moments we need to calculate it for each colloid
            # and then also modify the ODE to use the respective magmom instead the general
            self.colloid_mag_moment = np.array([col.magnetic_moment for col in self.colloids])
            self._calc_mag()
            self._calc_lub()

            if self.with_precalc_capillary:
                self._load_cap_forces()
            
            if self.save_h5:
                self._init_h5_output()

            self.integration_initialised = True


        state_flat = self._get_state_from_rafts().flatten() # (num_rafts, 3 (x,y,alpha) )

        if not isinstance(model, GlobalForceFunction):
            raise ValueError("Model must be of type GlobalForceFunction")

        for i in range(n_slices):       
            self.current_action = self.convert_actions_to_sim_units(
                model.calc_action(self.colloids)
            )

            logger.debug(f"{self.current_action=}")     
            
            for j in range(self.timesteps_per_slice):
                # Note: small numerical difference compared to paper due to sin(pi/2) != cos(0) exactly
                b_field = calc_B_field(self.current_action, self.time)
                self.current_action.magnetic_field = b_field

                mag_strength = np.sqrt(b_field[0] ** 2 + b_field[1] ** 2)
                mag_dir = np.arctan2(b_field[1], b_field[0])*(180/np.pi) % 360 #for some reason use degrees

                sol = scipy.integrate.solve_ivp(
                    self.RHS,
                    (self.time, self.time+self.params.time_step),
                    state_flat,
                    args=(mag_dir, mag_strength),
                    method="RK45",
                    #vectorized=False
                )

                self.time += self.params.time_step

                state_flat = sol.y[:, -1]
                state = state_flat.reshape((-1, 3))
                state[:, :2] = np.clip(state[:, :2], 0, self.params.box_length)

                if self.save_h5 and (j % self.timesteps_per_snapshot) == 0:
                    self.write_to_h5(sol.y[:, 0:1], sol.t[0:1], self.current_action)
                    model.save_agents()
            
            self._update_rafts_from_state(state)

            logger.debug(f"{sol.message=}")
            if not sol.success:
                # If we encounter an error, also save the state
                self.finalize_h5()
                raise RuntimeError(
                    f"Integration crashed at time {self.time}. Reason: {sol.message}"
                )
        self.finalize_h5()
        

    def convert_actions_to_sim_units(self, action: np.ndarray) -> MPIAction:
        Q_ = self.params.ureg.Quantity

        return MPIAction(
            magnitude=Q_(action[:2], "millitesla").m_as("sim_magnetic_field"),
            frequency=Q_([action[2], action[3]], "1 / second").m_as("sim_angular_velocity"),
            phase=[action[4], action[5]],
            offset=Q_(action[6:8], "millitesla").m_as("sim_magnetic_field")
        )

@numba.jit(fastmath=True, cache=True)
def compute_derivatives_numba(
    t, state_flat, num_rafts, mag_dir_deg, mag_strength,
    R, lub_threshold, fluid_density, dyn_viscosity,
    magnetic_constant, mag_moments,
    cm, cc, ch, tb, tm, tc,
    mag_dp_torque, mag_dp_force_on, mag_dp_force_off,
    cap_force, cap_torque,
    lubA, lubB, lubC, lubG, lub_coeff_scale,
    box_length, wall_repulsion_strength, force_due_to_curvature
):
    """
    This function should be outside of classes because numba does not
    work well with class objects. Rather create a wrapper inside the class
    that calls this functions with all relevant arguments.
    """
    # Pre-calculations
    pi_miu_R = np.pi * dyn_viscosity * R
    center_of_arena = box_length / 2.0
    
    # Reshape state
    raft_loc_x = state_flat[0::3]
    raft_loc_y = state_flat[1::3]
    raft_orient = state_flat[2::3]

    # Initialize accumulation arrays
    drdt = np.zeros((num_rafts, 2))
    dalphadt = np.zeros(num_rafts)
    raft_spin_speeds = np.zeros(num_rafts)

    mag_field_torque_term = np.zeros(num_rafts)
    
    # Calculate Torques and Spin Speeds
    for i in range(num_rafts):
        rx_i = raft_loc_x[i]
        ry_i = raft_loc_y[i]
        orient_i = raft_orient[i]
        
        # Magnetic Field Torque
        angle_diff = np.radians(mag_dir_deg - orient_i)
        mag_field_torque = mag_strength * mag_moments[i] * np.sin(angle_diff)
        
        # Base term
        term_mag_field = tb * mag_field_torque / (8 * pi_miu_R * R**2)
        
        term_mag_dipole = 0.0
        term_capillary = 0.0
        
        rji_ee_dist_smallest = R 

        for j in range(num_rafts):
            if i == j:
                continue
            
            rx_j = raft_loc_x[j]
            ry_j = raft_loc_y[j]
            
            dx = rx_i - rx_j
            dy = ry_i - ry_j
            dist_sq = dx*dx + dy*dy
            rji_norm = np.sqrt(dist_sq)
            rji_ee_dist = rji_norm - 2 * R
            
            # Angle calculation
            phi_ji = (np.degrees(np.arctan2(dy, dx)) - orient_i) % 360
            
            # Integer indices for dist and angle arrays
            idx_dist = int(rji_ee_dist + 0.5)
            idx_phi = int(phi_ji + 0.5)
            
            # Mag Dipole Torque
            if lub_threshold <= rji_ee_dist < 10000:
                term_mag_dipole += tm * mag_dp_torque[idx_dist, idx_phi] / (8 * pi_miu_R * R**2)
            elif 0 <= rji_ee_dist < lub_threshold:
                lub_idx = int(rji_ee_dist * lub_coeff_scale)
                term_mag_dipole += tm * lubG[lub_idx] * mag_dp_torque[idx_dist, idx_phi] / dyn_viscosity
            elif rji_ee_dist < 0:
                # Overlap case
                term_mag_dipole += tm * lubG[0] * mag_dp_torque[0, idx_phi] / dyn_viscosity

            # Capillary Torque
            if lub_threshold <= rji_ee_dist < 1000:
                term_capillary += tc * cap_torque[idx_dist, idx_phi] / (8 * pi_miu_R * R**2)
            elif 0 <= rji_ee_dist < lub_threshold:
                lub_idx = int(rji_ee_dist * lub_coeff_scale)
                term_capillary += tc * lubG[lub_idx] * cap_torque[idx_dist, idx_phi] / dyn_viscosity
            elif rji_ee_dist < 0:
                term_capillary += tc * lubG[0] * cap_torque[0, idx_phi] / dyn_viscosity
            
            # Lubrication Correction for Mag Field Torque
            if rji_ee_dist < lub_threshold and rji_ee_dist < rji_ee_dist_smallest:
                rji_ee_dist_smallest = rji_ee_dist
                
                if rji_ee_dist_smallest >= 0:
                    lub_idx = int(rji_ee_dist_smallest * lub_coeff_scale)
                    term_mag_field = (lubG[lub_idx] * mag_field_torque / dyn_viscosity) + \
                                     (lubC[lub_idx] * mag_dp_force_off[idx_dist, idx_phi] / dyn_viscosity)
                else:
                    term_mag_field = (lubG[0] * mag_field_torque / dyn_viscosity) + \
                                     (lubC[0] * mag_dp_force_off[idx_dist, idx_phi] / dyn_viscosity)

        # Store calculated terms
        mag_field_torque_term[i] = term_mag_field
        raft_spin_speeds[i] = term_mag_field + term_mag_dipole + term_capillary
    
    # Calculate Forces
    for i in range(num_rafts):
        rx_i = raft_loc_x[i]
        ry_i = raft_loc_y[i]
        orient_i = raft_orient[i]
        omega_i = raft_spin_speeds[i]
        
        fx = 0.0
        fy = 0.0
        
        # Curvature force
        if force_due_to_curvature != 0:
            # ri_center = centerOfArena - ri
            cx = center_of_arena - rx_i
            cy = center_of_arena - ry_i
            factor = force_due_to_curvature / (6 * pi_miu_R * (box_length/2))
            fx += factor * cx
            fy += factor * cy

        # Boundary lift force
        if num_rafts > 2:
            d_left = rx_i
            d_right = box_length - rx_i
            d_bottom = ry_i
            d_top = box_length - ry_i
            
            boundary_factor = fluid_density * (omega_i**2) * (R**7) / (6 * pi_miu_R)
            
            fx += boundary_factor * (1/d_left**3 - 1/d_right**3)
            fy += boundary_factor * (1/d_bottom**3 - 1/d_top**3)

        # Interactions with neighbors
        for j in range(num_rafts):
            if i == j:
                continue

            rx_j = raft_loc_x[j]
            ry_j = raft_loc_y[j]
            dx = rx_i - rx_j
            dy = ry_i - ry_j
            dist_sq = dx*dx + dy*dy
            rji_norm = np.sqrt(dist_sq)
            rji_ee_dist = rji_norm - 2 * R
            
            # Unit vectors
            ux = dx / rji_norm
            uy = dy / rji_norm
            # Cross product Z (y, -x)
            ux_cross_z = uy
            uy_cross_z = -ux
            
            phi_ji = (np.degrees(np.arctan2(dy, dx)) - orient_i) % 360
            idx_dist = int(rji_ee_dist + 0.5)
            idx_phi = int(phi_ji + 0.5)
            
            omega_j = raft_spin_speeds[j]

            # Mag Dipole Force On Axis
            term_val = 0.0
            if lub_threshold <= rji_ee_dist < 10000:
                term_val = cm * mag_dp_force_on[idx_dist, idx_phi] / (6 * pi_miu_R)
            elif 0 <= rji_ee_dist < lub_threshold:
                lub_idx = int(rji_ee_dist * lub_coeff_scale)
                term_val = cm * lubA[lub_idx] * mag_dp_force_on[idx_dist, idx_phi] / dyn_viscosity
            elif rji_ee_dist < 0:
                term_val = cm * lubA[0] * mag_dp_force_on[0, idx_phi] / dyn_viscosity
            
            fx += term_val * ux
            fy += term_val * uy
            
            # Capillary Force
            term_val = 0.0
            if lub_threshold <= rji_ee_dist < 1000:
                term_val = cc * cap_force[idx_dist, idx_phi] / (6 * pi_miu_R)
            elif 0 <= rji_ee_dist < lub_threshold:
                lub_idx = int(rji_ee_dist * lub_coeff_scale)
                term_val = cc * lubA[lub_idx] * cap_force[idx_dist, idx_phi] / dyn_viscosity
            elif rji_ee_dist < 0:
                term_val = cc * lubA[0] * cap_force[0, idx_phi] / dyn_viscosity
                
            fx += term_val * ux
            fy += term_val * uy

            # Hydrodynamic Force
            if rji_ee_dist >= lub_threshold:
                factor = ch * fluid_density * (omega_j**2) * (R**7) / (rji_norm**4 * 6 * pi_miu_R)
                fx += factor * dx # dx is rji vector
                fy += factor * dy
            elif 0 <= rji_ee_dist < lub_threshold:
                lub_idx = int(rji_ee_dist * lub_coeff_scale)
                factor = ch * lubA[lub_idx] * (fluid_density * (omega_j**2) * (R**7) / rji_norm**3) / dyn_viscosity
                fx += factor * ux
                fy += factor * uy

            # Wall Repulsion (Overlap)
            if rji_ee_dist < 0:
                factor = wall_repulsion_strength * (-rji_ee_dist / R) / (6 * pi_miu_R)
                fx += factor * ux
                fy += factor * uy

            # Mag Dipole Off Axis
            term_val = 0.0
            if lub_threshold <= rji_ee_dist < 10000:
                term_val = mag_dp_force_off[idx_dist, idx_phi] / (6 * pi_miu_R)
            elif 0 <= rji_ee_dist < lub_threshold:
                lub_idx = int(rji_ee_dist * lub_coeff_scale)
                term_val = lubB[lub_idx] * mag_dp_force_off[idx_dist, idx_phi] / dyn_viscosity
            elif rji_ee_dist < 0:
                term_val = lubB[0] * mag_dp_force_off[0, idx_phi] / dyn_viscosity
                
            fx += term_val * ux_cross_z
            fy += term_val * uy_cross_z

            # Velocity Torque Coupling / Mag Field Coupling
            mag_field_torque_val = mag_strength * mag_moments[i] * np.sin(np.radians(mag_dir_deg - orient_i))
            
            if rji_ee_dist >= lub_threshold:
                # velocity torque coupling
                factor = - (R**3) * omega_j / (rji_norm**2)
                fx += factor * ux_cross_z
                fy += factor * uy_cross_z
            elif 0 <= rji_ee_dist < lub_threshold:
                lub_idx = int(rji_ee_dist * lub_coeff_scale)
                torque_sum = mag_field_torque_val + mag_dp_torque[0, idx_phi] + cap_torque[idx_dist, idx_phi]
                factor = lubC[lub_idx] * torque_sum / dyn_viscosity
                fx += factor * ux_cross_z
                fy += factor * uy_cross_z
            elif rji_ee_dist < 0:
                torque_sum = mag_field_torque_val + mag_dp_torque[0, idx_phi] + cap_torque[idx_dist, idx_phi]
                factor = lubC[0] * torque_sum / dyn_viscosity
                fx += factor * ux_cross_z
                fy += factor * uy_cross_z

            # TODO: Here WCA implementation

        drdt[i, 0] = fx
        drdt[i, 1] = fy
    
    # Finalize dAlpha
    dalphadt = raft_spin_speeds / np.pi * 180.0
    
    # Flatten result
    state = np.empty(num_rafts * 3)
    state[0::3] = drdt[:, 0]
    state[1::3] = drdt[:, 1]
    state[2::3] = dalphadt
    
    return state