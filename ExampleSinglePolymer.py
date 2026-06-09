# ----------------------------------------------------------------------------
# Example File for Running Single-Polymer Analysis and Visualizaion Functions
# ----------------------------------------------------------------------------


"""
This script demonstrates a full analysis workflow for single-polymer simulation data,
including preprocessing, contact and distance analysis, genomic distance scaling, and
visualization of results.


Main analysis computations:
- Load and preprocess trajectory data (equilibration, subsampling, stride reduction)
- Compute and average contact maps across simulation runs
- Compute pairwise distance maps (average, per-run, and per-frame)
- Compute averaged contact and pairwise distance maps with their standard deviation
- Perform genomic distance scaling and compare with experimental data

Plots:
- Contact maps (per-run and averaged)
- Distance maps (average and frame-level)
- Average and standard deviation maps (contacts, pairwise distances)
- Genomic scaling plots (simulation vs experiment)

Designed for HPC execution on IDUN using GPUPolymerSimPipeline.

"""


import os
import sys
import numpy as np
import pandas as pd

sys.path.append("/cluster/work/andrhsta/Masteroppgave/Testing")

from GPUPolymerSimPipeline.Analysis.PolymerAnalysis import (
                                                            compute_average_contact_map,
                                                            average_run_contact_maps,
                                                            compute_average_distance_map,
                                                            average_run_distance_maps,
                                                            load_and_compute_genomic_stats,
                                                            compute_distance_map,
                                                            calc_distances
                                                        )

                                                            


from  GPUPolymerSimPipeline.Visualization.SinglePolymerVisualization import (
                                                                            plot_contact_map_vs_hic,
                                                                            plot_simulation_vs_tracing_half_half,
                                                                            plot_single_simulation_frames,
                                                                            plot_exp_vs_sim,
                                                                            plot_linear_and_distance_maps_large
                                                                        )



Trajectory_folder = "/cluster/work/andrhsta/Masteroppgave/Examples/Trajectory_A0.0_AB0.0_B0.32_coarsegrained_AE0.1"
Trajectory_analysis_folder = "/cluster/work/andrhsta/Masteroppgave/Examples/Trajectory_A0.0_AB0.0_B0.32_coarsegrained_AE0.1/Analysis"
Run_analysis_folder = "/cluster/work/andrhsta/Masteroppgave/Examples/Trajectory_A0.0_AB0.0_B0.32_coarsegrained_AE0.1/run_0_A=0.0_AB=0.0_B=0.32_attr=0.1_rep=3.0_steps=30000000_saveint=1000_eq=5000000_binsize=40000bp_cg=10000bp/Analysis"
Hi_c_path = "/cluster/work/andrhsta/Masteroppgave/Examples/GPUPolymerSimPipeline/ExperimentalData/GSE63525_IMR90_combined_90_500k.cool"

eq = 10000              # Frames to discard from initial conformations
jump = 1000             # Every Nth frame to include in analysis
stride = 20             # Every Nth monomer extracted for analysis


# -----------------------------
# PREPROCESSING
# -----------------------------

def load_traj(run_path):
    """
    Load coords_over_time for a single run.
    Replace this with your actual loader.
    Expected shape: (frames, N, 3)
    """
    return np.load(os.path.join(run_path, "trajectory_full.npy"))



def preprocess(coords, eq=eq, jump=jump, stride = stride):
    coords = coords[eq::jump]
    coords = coords[:, ::stride, :]
    return coords



# ------------------------------------------------------------
#  CONTACT MAPS
# ------------------------------------------------------------

run_folders = sorted(
    d for d in os.listdir(Trajectory_folder)
    if d.startswith("run_")
)

for run in run_folders:
    run_path = os.path.join(Trajectory_folder, run)

    coords_over_time = load_traj(run_path)   # you must implement this
    coords_over_time = preprocess(coords_over_time)

    compute_average_contact_map(
        coords_over_time,
        trajectory_path=run_path,
        cutoff= 12, 
    )


grand_avg_map, grand_std_map = average_run_contact_maps(
    Trajectory_folder
)



print("Plotting contact map vs Hi-C...")

plot_contact_map_vs_hic(
    Trajectory_analysis_folder,
    Hi_c_path,
    save_path=os.path.join(
        Trajectory_analysis_folder,
        "contact_vs_hic.png"
    ))




# ------------------------------------------------------------
#  PAIRWISE DISTANCE MAPS
# ------------------------------------------------------------

for run in run_folders:
    run_path = os.path.join(Trajectory_folder, run)

    coords_over_time = load_traj(run_path)
    coords_over_time = preprocess(coords_over_time)

    compute_average_distance_map(
        coords_over_time,
        trajectory_path=run_path
    )


grand_dist_avg_map, grand_dist_std_map = average_run_distance_maps(
    Trajectory_folder
)

print("Plotting simulation vs tracing...")

plot_simulation_vs_tracing_half_half(Trajectory_analysis_folder)



# ------------------------------------------------------------
#  SINGLE FRAMES PAIRWISE DISTANCE MAPS
# ------------------------------------------------------------

run_example = os.path.join(Trajectory_folder, run_folders[0])
coords = preprocess(load_traj(run_example))


print("Plotting single frames pairwise distances...")

dist_map = compute_distance_map(
    coords[0],
    trajectory_path=run_example,
    save_path="distance_map_frame1.npy"
)


plot_single_simulation_frames(Run_analysis_folder)


# ------------------------------------------------------------
#  AVERAGED AND STANDARD DEVIATION MAPS
# ------------------------------------------------------------


print("Plotting averaged and standard deviation maps...")

plot_linear_and_distance_maps_large(Trajectory_analysis_folder, save_path=os.path.join(Trajectory_analysis_folder, "linear_and_distance_maps.png"))




# ------------------------------------------------------------
#  GENOMIC DISTANCE SCALING 
# ------------------------------------------------------------

results, global_stats = load_and_compute_genomic_stats(
        main_folder=Trajectory_folder,
        bead_size_kb=10,
        eq=eq,
        jump=jump,
        stride=4,
        save_name="df_stats.csv"
    )


traces = pd.read_csv(
    "/cluster/work/andrhsta/Masteroppgave/Examples/GPUPolymerSimPipeline/ExperimentalData/combined_dataset_v2.csv.gz"
)

traces['tmp'] = (
    traces['exp']
    + '_'
    + traces['region']
    + '_'
    + traces['trace_id'].astype(str)
)

trace_id_map = {
    tmp: i for i, tmp in enumerate(traces.tmp.unique())
}

traces['trace_id'] = traces['tmp'].map(trace_id_map)

exp_df = calc_distances(
    traces=traces,
    region='Chr2_full',
    cell_type='IMR90_ctrl',
    crop_gpos=True,
    min_length=30
)


print("Plotting distance scaling curve...")

plot_exp_vs_sim(
    exp_df=exp_df,
    sim_csv_paths=[
        os.path.join(Trajectory_analysis_folder, "df_stats.csv")
    ],
    outpath=os.path.join(Trajectory_analysis_folder, "exp_vs_sim.png"),
    labels=["Simulation"],
    colors=["C0"],
    y_scale=20.0
)


print("\n All plots generated successfully!")