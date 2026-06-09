"""
-------------------------------------------------------------------------
Run Multiple Simulations with Identical Parameters and Save Trajectories
------------------------------------------------------------------------

Run multiple independent polymer simulations with identical parameters, either for a single-polymer
or a multiple-polymer system.

Each run performs a  simulation of a heteropolymer system and saves the 
full trajectory, parameters, and sequence data in a structured output folder.

"""


# Import modules
import os
import numpy as np
import sys
import json

import matplotlib.pyplot as plt


# For IDUN: Loading path to GPUPolymerSimPipeline and polychrom
sys.path.append('/cluster/work/andrhsta/Masteroppgave')
sys.path.append('/cluster/work/andrhsta/Masteroppgave/GPUPolymerSimPipeline')

# Importing polychrom modules:
from polychrom import simulation, starting_conformations, forcekits, forces

# Importing polymer sequence from sequence module
from GPUPolymerSimPipeline.Sequence.LoadingSequence import load_full_resolution_csv




# ---------------------
# Single-Polymer System
# ---------------------

def simulate_polymer_trajectories_replicates(
    base_params, 
    coarse_graining_bp, 
    n_runs=2, 
    output_folder="polymer_trajectories",
    bin_size_bp=40000,
    sequence=None
    ):

    
    """
    Run coarse-grained polymer simulations and save full trajectories.

    This function performs multiple independent stochastic simulations of a single
    coarse-grained heteropolymer model, where a base polymer sequence is
    expanded from a 40kb bins representation into a coarser bead
    representation of chosen size. 
    
    Each simulation run generates a time series of polymer
    configurations, which are saved as NumPy arrays along with metadata.

    Parameters
    ----------
    base_params : dict
        Dictionary containing all simulation parameters, including:
        - num_steps : int, total simulation steps
        - eq : int, amount of steps to be discarded before reaching equilibration
        - save_interval : int, steps between trajectory saves 
        - interaction_dict : dict with keys "A", "B", "AB",
                             giving interactionenergies between different monomer types
        - attractionEnergy, repulsionEnergy, repulsionRadius,
          attractionRadius, selectiveRepulsionEnergy,
          selectiveAttractionEnergy
        - other simulation-specific settings if needed



        Example: 
                    base_params = {
                                    "num_steps": 30_000_000,
                                    "save_interval": 1000,
                                    "eq": 10_000_000,
                                    "interaction_dict": {"A": 0.0, "AB": 0.0, "B": 0.0},
                                    "repulsionEnergy": 3.0,
                                    "repulsionRadius": 1.0,
                                    "attractionEnergy": 0.0,
                                    "attractionRadius": 1.5,
                                    "selectiveRepulsionEnergy": 0.0,
                                    "selectiveAttractionEnergy": 1.0       
                                   }

                                   

    bin_size_bp : int
        Size of the original sequence bin in base pairs before coarse-graining.

    coarse_graining_bp : int
        Resolution of coarse-grained beads in base pairs per monomer.

    n_runs : int, optional
        Number of independent simulation runs to perform (default is 2).

    output_folder : str, optional
        Directory where simulation outputs (trajectory and metadata) are saved.

    sequence : array-like of str
        Original polymer sequence at bin resolution, given as a list containing "0" or "1".
        This is expanded to a coarse-grained monomer sequence.

    Outputs
    -------
    For each run, a folder is created containing:
        - trajectory_full.npy : array of shape (timesteps, ...) with system state
        - base_params.json : simulation parameters used
        - sequence.json : coarse-grained monomer sequence

    Notes
    -----
    - A cubic lattice size is estimated from polymer length and a fixed volume
      fraction (0.3).
    - Polymer dynamics are simulated using a Langevin integrator on GPU.
    - No periodic boundary conditions are used (PBCbox=False).
    - The trajectory includes only saved snapshots (controlled by save_interval).
    """

    os.makedirs(output_folder, exist_ok=True)

    eq_start_block = base_params["eq"] // base_params["save_interval"]
    n_blocks = base_params["num_steps"] // base_params["save_interval"]
    density = 0.3  # polymer volume fraction

    if sequence is None:
        sequence = load_full_resolution_csv()

    # Coarse-grain sequence
    monomers_per_bin = bin_size_bp // coarse_graining_bp
    coarse_sequence = np.repeat(sequence, monomers_per_bin)
    N = len(coarse_sequence)

    # Compute cubic lattice size
    total_volume = N / density
    cubic_size = int(np.ceil(total_volume ** (1/3)))

    # Create parameter string
    param_str = (
        f"A={base_params['interaction_dict']['A']}_"
        f"AB={base_params['interaction_dict']['AB']}_"
        f"B={base_params['interaction_dict']['B']}_"
        f"attr={base_params['attractionEnergy']}_"
        f"rep={base_params['repulsionEnergy']}_"
        f"steps={base_params['num_steps']}_"
        f"saveint={base_params['save_interval']}_"
        f"eq={base_params['eq']}_"
        f"binsize={bin_size_bp}bp_"
        f"cg={coarse_graining_bp}bp"
    )

    for run_idx in range(n_runs):
        print(f"\nSimulation run {run_idx+1}/{n_runs}")
        run_folder = os.path.join(output_folder, f"run_{run_idx}_{param_str}")
        os.makedirs(run_folder, exist_ok=True)

        # Initialize simulation
        sim = simulation.Simulation(
            platform='CUDA',
            integrator='variableLangevin',
            error_tol=0.003,
            collision_rate=0.03,
            N=N,
            max_Ek=20.0,
            save_decimals=2,
            PBCbox=False,
            reporters=[]  # no HDF5
        )

        # Grow polymer
        polymer = starting_conformations.grow_cubic(N, cubic_size)
        sim.set_data(polymer, center=True)

        interaction_dict = base_params["interaction_dict"]

        # Add polymer chain forces
        sim.add_force(
            forcekits.polymer_chains(
                sim,
                chains=[(0, None, False)],
                bond_force_func=forces.harmonic_bonds,
                bond_force_kwargs={"bondLength":1.0, "bondWiggleDistance":0.1},
                angle_force_func=forces.angle_force,
                angle_force_kwargs={"k":1.5},
                nonbonded_force_func=None,
                nonbonded_force_kwargs={"trunc":3.0},
                except_bonds=True
            )
        )

        # Add heteropolymer interactions
        sim.add_force(
            forces.heteropolymer_SSW(
                sim,
                interactionMatrix=np.array([
                    [interaction_dict["A"], interaction_dict["AB"]],
                    [interaction_dict["AB"], interaction_dict["B"]]
                ]),
                monomerTypes=coarse_sequence,
                extraHardParticlesIdxs=[],
                repulsionEnergy=base_params["repulsionEnergy"],
                repulsionRadius=base_params["repulsionRadius"],
                attractionEnergy=base_params["attractionEnergy"],
                attractionRadius=base_params["attractionRadius"],
                selectiveRepulsionEnergy=base_params["selectiveRepulsionEnergy"],
                selectiveAttractionEnergy=base_params["selectiveAttractionEnergy"],
                name="heteropolymer_SSW"
            )
        )

        # Run simulation and collect trajectory
        trajectory = []
        for _ in range(n_blocks):
            sim.do_block(base_params["save_interval"])
            trajectory.append(sim.get_data())

        # Convert to numpy array in default dtype (float64)
        trajectory = np.array(trajectory)  # keep float64



        # Save trajectory, base params, and sequence
        np.save(os.path.join(run_folder, "trajectory_full.npy"), trajectory)

        with open(os.path.join(run_folder, "base_params.json"), "w") as f:
            json.dump(base_params, f, indent=4)

        with open(os.path.join(run_folder, "sequence.json"), "w") as f:
            json.dump(coarse_sequence.tolist(), f, indent=4)

        print(f"Run {run_idx+1} completed and saved in {run_folder}")








# --------------------
# Multi-Polymer System
# --------------------


POLYMER_COLORS = [
    ("green",   "#2ca02c", "#98df8a"),
    ("blue",    "#1f77b4", "#aec7e8"),
    ("red",     "#d62728", "#ff9896"),
    ("purple",  "#9467bd", "#c5b0d5"),
    ("orange",  "#ff7f0e", "#ffbb78"),
    ("brown",   "#8c564b", "#c49c94"),
    ("cyan",    "#17becf", "#9edae5"),
]

def get_polymer_colors(p):
    """Return (color_A, color_B) for polymer p"""
    base, dark, light = POLYMER_COLORS[p % len(POLYMER_COLORS)]
    return light, dark  # A, B




def compute_confinement_radius(N, density):
    return ((3 * N) / (4 * np.pi * density)) ** (1/3)




def draw_sphere(ax, radius, alpha=0.1, color="gray"):
    u = np.linspace(0, 2*np.pi, 50)
    v = np.linspace(0, np.pi, 50)

    x = radius * np.outer(np.cos(u), np.sin(v))
    y = radius * np.outer(np.sin(u), np.sin(v))
    z = radius * np.outer(np.ones_like(u), np.cos(v))

    ax.plot_surface(x, y, z, color=color, alpha=alpha, linewidth=0)




def simulate_polymer_trajectories_multiple_polymers_replicates(
    base_params,
    bin_size_bp,
    coarse_graining_bp,
    n_polymers,
    n_runs=1,
    output_folder="polymer_trajectories_3polymers",
    sequence=None,
):
    
    """
    Run simulations of multiple interacting heteropolymer chains
    confined in a spherical volume, with optional replicate runs.

    This function:
      - Coarse-grains a binary polymer sequence (A/B monomers)
      - Replicates it across multiple polymer chains
      - Places polymers in a symmetric 3D arrangement inside a confinement sphere
      - Simulates dynamics using a GPU-accelerated Variable Langevin integrator
      - Includes:
          * harmonic bond constraints along each chain
          * angle stiffness
          * sequence-dependent heteropolymer interactions (SSW model)
          * spherical confinement potential
      - Saves trajectories, parameters, and sequence data
      - Produces initial and final 3D structure visualizations

    Parameters
    ----------
    base_params : dict
        Dictionary of simulation parameters. Expected keys include:
        - num_steps : int
        - save_interval : int
        - num_polymers: int, NB!!! new num_polymers needs new initial placements 
        - density : float
        - interaction_dict : dict with keys {"A", "B", "AB"}
        - repulsionEnergy, repulsionRadius
        - attractionEnergy, attractionRadius
        - selectiveRepulsionEnergy, selectiveAttractionEnergy

    bin_size_bp : int
        Genomic bin size used for coarse-graining (base pairs per bin).

    coarse_graining_bp : int
        Resolution of the input sequence (base pairs per monomer before binning).

    n_polymers : int
        Number of identical polymer chains simulated in parallel.

    n_runs : int, optional
        Number of independent simulation replicates (default = 1).

    output_folder : str, optional
        Root directory where all simulation outputs are saved.

    sequence : array-like or None
        Binary A/B sequence (0/1).

    Outputs (per run)
    ------------------
    Saved in `output_folder/run_i_*`:
      - trajectory.npy : (timesteps, N, 3) polymer coordinates
      - params.json : simulation parameters
      - sequence.json : polymer sequences per chain
      - initial_structure.png : starting configuration
      - lastframe_structure.png : final configuration

    Notes
    -----
    - Polymers are arranged in a star-like geometry (center + ±x, ±y, ±z offsets).
    - Confinement radius is computed from density via `compute_confinement_radius`.
    - Polymer separation is dynamically set from confinement size.
    - Total particle number is:
          N = n_polymers x sequence_length

    """

    os.makedirs(output_folder, exist_ok=True)

    n_blocks = base_params["num_steps"] // base_params["save_interval"]

    if sequence is None:
        sequence = load_full_resolution_csv()

    # Sequence processing (A/B alternating or provided)
    monomers_per_bin = bin_size_bp // coarse_graining_bp
    coarse_sequence = np.repeat(sequence, monomers_per_bin)
    coarse_sequence = np.asarray(coarse_sequence, dtype=int)

    cubic_density = 0.3
    L = len(coarse_sequence)
    N = n_polymers * L
    density = base_params["density"]

    R_conf = compute_confinement_radius(N, density)
    base_params["confinement_radius"] = R_conf

    # your definition
    base_params["polymer_separation"] = 84.42770069476909 # Samme som for sph0.01
    base_params["polymer_separation"] = (2 * R_conf) / 3

    total_volume = N / cubic_density
    cubic_size = int(np.ceil(total_volume ** (1/3)))

    param_str = f"3polymers_A{base_params['interaction_dict']['A']}_B{base_params['interaction_dict']['B']}"


    # Runs
    for run_idx in range(n_runs):

        print(f"\nRun {run_idx+1}/{n_runs}")

        run_folder = os.path.join(output_folder, f"run_{run_idx}_{param_str}")
        os.makedirs(run_folder, exist_ok=True)



        # Simulation
        sim = simulation.Simulation(
            platform='CUDA',
            integrator='variableLangevin',
            error_tol=0.003,
            collision_rate=0.03,
            N=N,
            max_Ek=20.0,
            save_decimals=2,
            PBCbox=False,
            reporters=[]
        )


        # Setup for 7-polymer system
        positions = []

        spacing = base_params["polymer_separation"]

        # Define relative positions (center + 6 directions)
        offsets = [
            (0, 0, 0),   # center

            (+1, 0, 0),  # +x
            (-1, 0, 0),  # -x

            (0, +1, 0),  # +y
            (0, -1, 0),  # -y

            (0, 0, +1),  # +z
            (0, 0, -1),  # -z
        ]

        for dx, dy, dz in offsets:

            chain = starting_conformations.grow_cubic(L, cubic_size).astype(float)

            shift = np.array([dx * spacing, dy * spacing, dz * spacing])
            chain += shift

            positions.append(chain)

        positions = np.vstack(positions)
        sim.set_data(positions, center=True)
        initial_positions = sim.get_data().copy()



        # Bonds
        chains = []
        for p in range(n_polymers):
            chains.append((p * L, (p + 1) * L, False))

        sim.add_force(
            forcekits.polymer_chains(
                sim,
                chains=chains,
                bond_force_func=forces.harmonic_bonds,
                bond_force_kwargs={"bondLength": 1.0, "bondWiggleDistance": 0.1},
                angle_force_func=forces.angle_force,
                angle_force_kwargs={"k": 1.5},
                nonbonded_force_func=None,
                except_bonds=True
            )
        )

        # Heteropolymer interactions
        full_sequence = np.concatenate(
            [coarse_sequence for _ in range(n_polymers)]
        )

        interaction_dict = base_params["interaction_dict"]

        sim.add_force(
            forces.heteropolymer_SSW(
                sim,
                interactionMatrix=np.array([
                    [interaction_dict["A"], interaction_dict["AB"]],
                    [interaction_dict["AB"], interaction_dict["B"]]
                ]),
                monomerTypes=full_sequence,
                extraHardParticlesIdxs=[],
                repulsionEnergy=base_params["repulsionEnergy"],
                repulsionRadius=base_params["repulsionRadius"],
                attractionEnergy=base_params["attractionEnergy"],
                attractionRadius=base_params["attractionRadius"],
                selectiveRepulsionEnergy=base_params["selectiveRepulsionEnergy"],
                selectiveAttractionEnergy=base_params["selectiveAttractionEnergy"],
                name="heteropolymer_SSW"
            )
        )


        # Initial configuration visualization 
        init_pos = initial_positions - np.mean(initial_positions, axis=0)

        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')


        # Add visualization of confining sphere if defined
        R = R_conf

        draw_sphere(ax, R, alpha=0.15, color="lightblue")

        for p in range(n_polymers):

            A_color, B_color = get_polymer_colors(p)

            start = p * L
            end = (p + 1) * L

            chain = init_pos[start:end]
            seq = full_sequence[start:end]

            for i in range(len(chain) - 1):
                color = A_color if seq[i] == 0 else B_color
                ax.plot(chain[i:i+2, 0],
                        chain[i:i+2, 1],
                        chain[i:i+2, 2],
                        color=color,
                        linewidth=2)

            bead_colors = [A_color if s == 0 else B_color for s in seq]

            ax.scatter(chain[:, 0], chain[:, 1], chain[:, 2],
                    c=bead_colors, s=40, edgecolors='black')

        ax.set_title("Initial polymer placement inside confinement sphere")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

        ax.set_xlim(-120, 120)
        ax.set_ylim(-120, 120)
        ax.set_zlim(-120, 120)

        ax.view_init(20, 45)

        plt.tight_layout()
        plt.savefig(os.path.join(run_folder, "initial_structure.png"), dpi=300)
        plt.close()



        # Run simulation
        trajectory = []
        

        # Add spherical confinement

        sim.add_force(
            forces.spherical_confinement(
                sim,
                density=density,  # target density inside the sphere
                k=5.0,          # strength of confinement (increase = tighter sphere)
                name="spherical_confinement"
            )
        )
        

        for _ in range(n_blocks):
            sim.do_block(base_params["save_interval"])
            trajectory.append(sim.get_data())

        trajectory = np.array(trajectory)

        

        # Save
        np.save(os.path.join(run_folder, "trajectory.npy"), trajectory)

        with open(os.path.join(run_folder, "params.json"), "w") as f:
            json.dump(base_params, f, indent=4)


        sequence_data = {
            "n_polymers": n_polymers,
            "L": L,
            "monomers_per_bin": monomers_per_bin,
            "original_sequence": sequence.tolist() if isinstance(sequence, np.ndarray) else sequence,     
            "coarse_sequence": coarse_sequence.tolist(),   # per polymer
            "polymers": []
        }

        for p in range(n_polymers):
            start = p * L
            end = (p + 1) * L

            sequence_data["polymers"].append({
                "polymer_id": p,
                "sequence": full_sequence[start:end].tolist()
            })

        with open(os.path.join(run_folder, "sequence.json"), "w") as f:
                json.dump(sequence_data, f, indent=4)

        print(f"Saved: {run_folder}")

       

        # Final visualization
        final_positions = trajectory[-1]
        final_positions = final_positions - np.mean(final_positions, axis=0)

        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')

        for p in range(n_polymers):

            A_color, B_color = get_polymer_colors(p)

            start = p * L
            end = (p + 1) * L
            chain = final_positions[start:end]
            seq =full_sequence[start:end]

            # draw bonds
            for i in range(len(chain) - 1):
                color = A_color if seq[i] == 0 else B_color
                ax.plot(
                    chain[i:i+2, 0],
                    chain[i:i+2, 1],
                    chain[i:i+2, 2],
                    color=color,
                    linewidth=2
                )

            # draw beads
            bead_colors = [A_color if s == 0 else B_color for s in seq]

            ax.scatter(
                chain[:, 0],
                chain[:, 1],
                chain[:, 2],
                c=bead_colors,
                s=50,
                edgecolors='black'
            )


        ax.set_xlim(-120, 120)
        ax.set_ylim(-120, 120)
        ax.set_zlim(-120, 120)

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")

        ax.view_init(20, 45)

        plt.tight_layout()

        plt.savefig(os.path.join(run_folder, "lastframe_structure.png"), dpi=300)
        plt.close()
