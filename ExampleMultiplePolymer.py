# ----------------------------------------------------------------------------
# Example File for Running Multi-Polymer Analysis and Visualization Functions
# ----------------------------------------------------------------------------


"""

This script demonstrates a full analysis workflow for multi-polymer simulation data,
including preprocessing, contact and distance analysis, genomic distance scaling, and
visualization of results.

Main analysis computations:
- Multi-polymer contact maps giving intra- and interpolymer contacts
- Pairwise distance maps
- Polymer-polymer overlap statistics across selected polymers
- simulation scaling curves

Plots:
- Contact maps
- Distance maps comparing with tracing data
- Polymer overlap distributions
- Comparison of experimental and simulation distance scaling curves

Designed for HPC execution on IDUN using GPUPolymerSimPipeline.

"""

import os
import sys
import numpy as np
import pandas as pd

sys.path.append("/cluster/work/andrhsta/Masteroppgave/Testing")

from GPUPolymerSimPipeline.Analysis.PolymerAnalysis import (
                                                                compute_contact_map_multipolymer,
                                                                compute_and_save_multi_polymer_distance_maps,
                                                                load_and_compute_genomic_stats_multipolymers,
                                                                compute_polymer_overlap,
                                                                calc_distances
                                                            )

from GPUPolymerSimPipeline.Visualization.MultiPolymerVisualization import (
                                                                plot_contact_map,
                                                                plot_simulation_vs_tracing_half_polymeravg,
                                                                plot_sim_vs_tracing_half_map_allpolymeravg,
                                                                plot_polymer_overlap_boxplot_from_npy,
                                                                plot_exp_vs_sim,
                                                                plot_exp_single_multi_scaled
                                                            )





# -----------------------------
# SETTINGS
# -----------------------------

trajectory_folders = [
    "/cluster/work/andrhsta/Masteroppgave/Examples/[B0.34]_7_polymers_sph0.0__numSteps30000000_saveinterval5000_k=5.0"
]



run_name = "run_0_3polymers_A0.0_B0.34"
trajectory_file = "trajectory.npy"
sequence_file = "sequence.json"

bead_size_kb = 20

eq = 10000
jump = 1000
stride = 20

# -----------------------------
# PATHS
# -----------------------------

system_folder = trajectory_folders[0]

run_folder = os.path.join(
    system_folder,
    run_name,
)

traj_path = os.path.join(
    run_folder,
    trajectory_file,
)

seq_path = os.path.join(
    run_folder,
    sequence_file,
)

traj_files = [traj_path]

system_analysis = os.path.join(
    system_folder,
    "Analysis",
)

input_path = os.path.join(
    system_analysis,
    "polymer_overlap.npy",
)

# -----------------------------
# PREPROCESSING
# -----------------------------


def load_coords(run_folder):
    """
    Load coordinates for a single run.
    Expected shape: (frames, N, 3)
    """
    return np.load(os.path.join(run_folder, "coords.npy"))


def preprocess(coords, eq=eq, jump=jump, stride=stride):
    coords = coords[eq::jump]
    coords = coords[:, ::stride, :]
    return coords




# ----------------------------------------------------------------------------
# CONTACT MAPS 
# ----------------------------------------------------------------------------

compute_contact_map_multipolymer(
    traj_path=traj_path,
    seq_path=seq_path,
    cutoff=10.0,
    k=5,
    frame_start=2000,
    frame_stride=1000,
)

print("Plotting contact map...")

plot_contact_map(traj_path=traj_files[0])


# ----------------------------------------------------------------------------
# PAIRWISE DISTANCE MAPS
# ----------------------------------------------------------------------------

compute_and_save_multi_polymer_distance_maps(
    traj_file=traj_path,
    eq=2000,
    jump=1000,
)

print("Plotting global vs tracing...")

plot_simulation_vs_tracing_half_polymeravg(
    traj_files=traj_files
)

plot_sim_vs_tracing_half_map_allpolymeravg(
    run_folder=run_folder
)

# ----------------------------------------------------------------------------
# OVERLAP
# ----------------------------------------------------------------------------

system_name = (
    r"$E_{\mathrm{attr}}^{B-B} = 0.32,$\n"
    r"$\rho = 0.01$"
)

systems = {
            system_name: os.path.join(system_folder, "run_0_3polymers_A0.0_B0.34", "trajectory.npy")
}


label_dict = {
    system_name: {
        "Ae": 0.0,
        "B": 0.32,
        "rho": 0.01,
    }
}

compute_polymer_overlap(
    systems=systems,
    label_dict=label_dict,
    n_monomers=10,
    polymer_ids=[0, 1, 2, 3, 4, 5],
    equilibration=2000,
    jump=1000,
    output_path="polymer_overlap.npy",
)



print("Plotting polymer overlap boxplot...")

plot_polymer_overlap_boxplot_from_npy(
    systems=systems,
    label_dict=label_dict,
    input_path=input_path
)





# ----------------------------------------------------------------------------
# GENOMIC DISTANCE SCALING
# ----------------------------------------------------------------------------

for sys_folder in trajectory_folders:

    traj_file = os.path.join(
        sys_folder,
        run_name,
        trajectory_file,
    )

    load_and_compute_genomic_stats_multipolymers(
        traj_file=traj_file,
        bead_size_kb=bead_size_kb,
    )


global_csv = os.path.join(
    system_analysis,
    "sim_global_stats.csv"
)

per_polymer_csv = os.path.join(
    run_folder,
    "Analysis",
    "sim_per_polymer_stats.csv"
)

single_csv = (
    "/cluster/work/andrhsta/Masteroppgave/Examples/Trajectory_A0.0_AB0.0_B0.32_coarsegrained_AE0.1/Analysis/df_stats.csv"
)



print("Loading simulation data...")

global_df = pd.read_csv(global_csv)
multi_df = pd.read_csv(per_polymer_csv)
single_df = pd.read_csv(single_csv)

# standardize column names
global_df = global_df.rename(columns={"dist_mean": "dist"})
multi_df = multi_df.rename(columns={"dist_mean": "dist"})
single_df = single_df.rename(columns={"dist_mean": "dist"})


print("Plotting genomic distance comparison...")


# EXPERIMENTAL DATA

traces = pd.read_csv(
    "/cluster/work/andrhsta/Masteroppgave/Examples/GPUPolymerSimPipeline/ExperimentalData/combined_dataset_v2.csv.gz"
)

traces["tmp"] = (
    traces["exp"]
    + "_"
    + traces["region"]
    + "_"
    + traces["trace_id"].astype(str)
)

trace_id_map = {tmp: i for i, tmp in enumerate(traces.tmp.unique())}
traces["trace_id"] = traces["tmp"].map(trace_id_map)

exp_df = calc_distances(
    traces=traces,
    region="Chr2_full",
    cell_type="IMR90_ctrl",
    crop_gpos=True,
    min_length=30
)


# PLOTTING

# 1. EXP vs SIM
plot_exp_vs_sim(
    exp_df=exp_df,
    sim_csv_files=[global_csv],
    labels=["Simulation"],
    outpath=os.path.join(
        system_analysis,
        "genomic_distance_plot.png"
    ),
    x_scale=1.0,
    y_scale=20.0,
    x_cut=120_000,
)


# 2. SINGLE vs MULTI
plot_exp_single_multi_scaled(
    exp_df=exp_df,
    single_csv=single_csv,
    multi_csv=global_csv,
    single_scale=20.0,
    multi_scale=20.0,
    outpath=os.path.join(
        system_analysis,
        "single_vs_multi.png"
    ),
)






print("\nALL PLOTS COMPLETE\n")