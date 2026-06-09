"""
----------------------------------------------------------
Polymer Visualization Functions for Single-polymer Systems
----------------------------------------------------------

"""

# Importing modules
import os
import sys
import glob
import json
import cooler
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path


import matplotlib
matplotlib.use("Agg")

import matplotlib.colors as mcolors
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from skimage.transform import resize
from mpl_toolkits.axes_grid1 import make_axes_locatable


# For IDUN: Loading path to GPUPolymerSimPipeline and polychrom
sys.path.append('/cluster/work/andrhsta/Masteroppgave')

# For importing LoopTrace when using IDUN:
sys.path.append("/cluster/work/andrhsta/Masteroppgave/TracingData")
sys.path.append("/cluster/work/andrhsta/Masteroppgave/TracingData/TraceAnalysis")

from looptrace import trace_analysis_functions as tr # Custom functions for analyzing tracing data
from scipy.spatial.distance import pdist, squareform


# Importing polymer sequence from sequence module
from GPUPolymerSimPipeline.Analysis.PolymerAnalysis import (radius_of_gyration)
from GPUPolymerSimPipeline.ExperimentalData.ExperimentalData import (replace_nan_with_diagonal_median, calc_distances, round_to_nearest_multiple)


# ---------------
# Apply nice font
# ---------------

def apply_serif_font():
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["DejaVu Serif"]


apply_serif_font()



# ----------------------------------------
# Radius of Gyration over Simulation steps
# ----------------------------------------

def plot_Rg_vs_time_coords(
    trajectory_folder,
    output_file="Rg_vs_time.png",
    colors=[
        "#48864C",
        "#7AA3BA",
        "#6A0297",
        "#832E49",
        "#A39304"
    ]
):
    apply_serif_font()

    trajectory_folder = Path(trajectory_folder)

    if not trajectory_folder.exists():
        raise FileNotFoundError(f"Trajectory folder not found: {trajectory_folder}")

    plt.figure(figsize=(16, 9))
    ax = plt.gca()

    run_idx = 0

    # --------------------------------------------------
    # find only run_* folders
    # --------------------------------------------------
    run_folders = sorted([
        p for p in trajectory_folder.iterdir()
        if p.is_dir() and p.name.startswith("run_")
    ])

    for run_folder in run_folders:

        analysis_folder = run_folder / "Analysis"
        rg_file = analysis_folder / "rg.npy"
        params_file = run_folder / "base_params.json"

        if not rg_file.exists():
            print(f"Skipping missing: {rg_file}")
            continue

        if not params_file.exists():
            print(f"Skipping missing params: {params_file}")
            continue

        import json
        with open(params_file, "r") as f:
            params = json.load(f)

        save_interval = params["save_interval"]

        rg_values = np.load(rg_file)
        time_points = np.arange(len(rg_values)) * save_interval

        color = colors[run_idx % len(colors)]

        ax.plot(
            time_points,
            rg_values,
            lw=1.5,
            alpha=0.8,
            label=f"run {run_idx}",
            color=color
        )

        run_idx += 1

    ax.set_ylim(0, 120)
    ax.set_xlabel("Simulation step", fontsize=30)
    ax.set_ylabel(r"$\langle R_g^2 \rangle^{1/2}$", fontsize=30)
    ax.tick_params(axis='both', which='major', labelsize=28)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(fontsize=25)

    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close()






# -------------------------------------------
# Hi-C Data vs Simulated Averaged Contact Map
# --------------------------------------------

def plot_contact_map_vs_hic(
    trajectory_analysis_folder,
    hic_file,
    save_path=None,
    start_bin=0,
    end_bin=487,
    sim_vmin=1e-1,
    hic_vmin=0,
    hic_vmax=10
):
    """
    Compute global average contact map across multiple runs and plot with Hi-C map (half-and-half),
    resizing Hi-C to match simulation data dimensions.

    Upper triangle = simulation (log scale)
    Lower triangle = Hi-C (log1p, resized to simulation)

    Parameters
    ----------
    traj_files : list of str
        Paths to trajectory .npy files
    hic_file : str
        Path to Hi-C .cool file
    jump : int
        Frames to skip for averaging
    cutoff : float
        Cutoff distance for contacts
    save_path : str or None
        Path to save figure
    start_bin : int
        Start bin index for Hi-C subset
    end_bin : int
        End bin index for Hi-C subset
    sim_vmin : float
        Minimum value for simulation log scale
    hic_vmin : float or None
        Minimum value for Hi-C colorbar (lower threshold)
    hic_vmax : float or None
        Maximum value for Hi-C colorbar (upper threshold)
    """

    apply_serif_font()


    avg_file = os.path.join(
    trajectory_analysis_folder,
    "average_contact_map_all_runs.npy"
    )

    if not os.path.exists(avg_file):
        raise FileNotFoundError(
            f"Missing grand average contact map:\n{avg_file}"
        )

    global_avg_contact = np.load(avg_file)
    sim_shape = global_avg_contact.shape



    # Load Hi-C matrix
    c = cooler.Cooler(hic_file)
    hic_matrix = c.matrix(balance=False).fetch("2")
    hic_matrix = hic_matrix[start_bin:end_bin, start_bin:end_bin]

    # Resize Hi-C to match simulation shape
    hic_resized = resize(
        hic_matrix,
        global_avg_contact.shape,
        order=1,
        mode="reflect",
        anti_aliasing=True,
        preserve_range=True
    )

    # Half-and-half plot
    N = sim_shape[0]
    upper_mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    lower_mask = np.tril(np.ones((N, N), dtype=bool), k=-1)

    upper_triangle = np.where(upper_mask, global_avg_contact + 1e-5, np.nan)
    lower_triangle = np.where(lower_mask, np.log1p(hic_resized), np.nan)


    # Plot
    colors = [
                "#053061",  # dark blue
                "#2166AC",
                "#92C5DE",
                "#F7F7F7",  # white center
                "#DA8683",
                "#DC6472",
                "#F70404"   # dark red
            ]

    custom_cmap = mcolors.LinearSegmentedColormap.from_list(
                  "mycmap",
                  colors
                )

    fig, ax = plt.subplots(figsize=(10, 10))

    # Simulation (upper triangle) log scale
    sim_norm = LogNorm(vmin=sim_vmin, vmax=1)
    im_sim = ax.imshow(upper_triangle, cmap=custom_cmap, norm=sim_norm, origin="upper")

    # Hi-C (lower triangle) with optional vmax/vmin
    im_hic = ax.imshow(
        lower_triangle,
        cmap= custom_cmap,
        origin="upper",
        vmin=hic_vmin,
        vmax=hic_vmax
    )

    # Axes
    ax.set_xlabel("Bin Index / Monomer", fontsize=27)
    ax.set_ylabel("Bin Index / Monomer", fontsize=27)
    ax.tick_params(axis='both', which='major', labelsize=25)
    ax.tick_params(axis='both', which='minor', labelsize=25)

    # Simulation colorbar
    cbar_sim = fig.colorbar(im_sim, ax=ax, fraction=0.046, pad=0.04)
    cbar_sim.set_label("Simulation Avg Contacts (log scale)", fontsize=27)
    cbar_sim.set_ticks([1e-1, 1])
    cbar_sim.minorticks_off()
    cbar_sim.ax.tick_params(labelsize=25)

    # Hi-C colorbar underneath
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("bottom", size="5%", pad=0.9)
    cbar_hic = plt.colorbar(im_hic, cax=cax, orientation='horizontal')
    cbar_hic.set_label("Contact Count (log scale)", fontsize=27)
    cbar_hic.ax.tick_params(labelsize=25)

    plt.tight_layout()

    # Save or show
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()







# ------------------------------------------------------------------
# Chromatin Tracing Data vs Simulated Averaged Pairwise Distance Map
# ------------------------------------------------------------------



def plot_simulation_vs_tracing_half_half(
    traj_analysis_folder,
    B=0.32,
    AE=0.1,
):
    """
    Plot a half-and-half comparison between averaged simulation distance maps
    and DNA-tracing distance maps.

    Upper triangle = simulation average distances
    Lower triangle = DNA-tracing average distances
    """


    apply_serif_font()


    # Load DNA-tracing data
    traces = pd.read_csv(
        "/cluster/work/andrhsta/Masteroppgave/Examples/GPUPolymerSimPipeline/ExperimentalData/combined_dataset_v2.csv.gz",
        low_memory=False
    )

    traces['tmp'] = traces['exp'] + '_' + traces['region'] + '_' + traces['trace_id'].astype(str)
    trace_id_map = {tmp: i for i, tmp in enumerate(traces.tmp.unique())}
    traces['trace_id'] = traces['tmp'].map(trace_id_map)

    region = 'Chr2_full'
    cell_type = 'IMR90_ctrl'

    df = traces.loc[
        (traces.region == region) &
        (traces.cell_type == cell_type) &
        (traces.g_pos > 0)]
    full_count = 239 if region == 'Chr2_full' else 84

    valid_trace_ids = (
        df.groupby('trace_id').count()['fit_id'] == full_count
    )

    df = df[df.trace_id.isin(valid_trace_ids[valid_trace_ids].index)]
    df = tr.tracing_length_qc(df, min_length=30)

    pwds = tr.pwd_calc(df)
    obs_dists = np.nanmean(pwds, axis=0)
    obs_dists = replace_nan_with_diagonal_median(obs_dists)


    avg_file = os.path.join(
    traj_analysis_folder,
    "average_distance_map_all_runs.npy"
    )

    if not os.path.exists(avg_file):
        raise FileNotFoundError(
            f"Missing grand average distance map:\n{avg_file}"
        )

    sim_avg = np.load(avg_file)


    # Resize tracing matrix
    tracing_sub = obs_dists

    tracing_resized = resize(
        tracing_sub,
        sim_avg.shape,
        order=1,
        mode='reflect',
        anti_aliasing=True,
        preserve_range=True
    )

    # Half-and-half plot
    N = sim_avg.shape[0]

    upper_mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    lower_mask = np.tril(np.ones((N, N), dtype=bool), k=-1)

    upper_triangle = np.where(upper_mask, sim_avg + 1e-5, np.nan)
    lower_triangle = np.where(lower_mask,  tracing_resized + 1e-5, np.nan)

    fig, ax = plt.subplots(figsize=(15, 24))

    # Plot upper and lower triangles
    im_sim = ax.imshow(
        upper_triangle,
        cmap='RdBu',
        vmin=0,
        vmax=np.nanmax(sim_avg)
    )

    im_tracing = ax.imshow(
        lower_triangle,
        cmap='RdBu',
        vmin=200,
        vmax=3500
    )

 
    # Axis labels
    ax.set_xlabel("Monomer Index", fontsize=30)
    ax.set_ylabel("Monomer Index", fontsize=30)


    # Tick labels
    ax.tick_params(axis='both', which='major', labelsize=35)
    ax.tick_params( axis='both', which='minor', labelsize=35)

    # Colorbars
    divider = make_axes_locatable(ax)

    cax_sim = divider.append_axes("right", size="4%", pad=0.15)
    cbar_sim = plt.colorbar(im_sim, cax=cax_sim)
    cbar_sim.set_label("Simulation Distance (nm)", fontsize=35 )
    cbar_sim.ax.tick_params(labelsize=30)

    cax = divider.append_axes("bottom", size="4%", pad=1)

    cbar_tracing = plt.colorbar(im_tracing, cax=cax, orientation='horizontal')
    cbar_tracing.set_label("DNA-Tracing Distance (nm)", fontsize=35)
    cbar_tracing.ax.tick_params(labelsize=35)


    # Save figure
    output_path = os.path.join(
        traj_analysis_folder,
        (
            f"sim_vs_tracing_half_multi_run_"
            f"AE{AE}_B{B}.png"
        )
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)

    print(f"Half-and-half figure saved to {output_path}")







# ---------------------------------------
# Plotting Single Frames from Simulations
# ---------------------------------------



def plot_single_simulation_frames(
    traj_analysis_folder,
    B=0.3,
    AE=0.1,
):
    """
    Plot all precomputed distance matrix files for a single simulation.

    This function scans the provided trajectory folder for all NumPy files
    matching the pattern "dist_frames*.npy" (optionally with run or frame
    indices in the filename), loads them, and generates heatmap visualizations
    for each distance matrix found.
    """


    # Find all precomputed distance frame files
    pattern = os.path.join(traj_analysis_folder, "**", "distance_map_frame*.npy")
    frame_files = sorted(glob.glob(pattern, recursive=True))

    if len(frame_files) == 0:
        raise FileNotFoundError(
            f"No distance_map_frame*.npy files found in {traj_analysis_folder}"
        )

    print(f"Found {len(frame_files)} frame files")

    # Loop over all matching files
    for file_path in frame_files:

        data = np.load(file_path)

        # extract frame number from filename
        base = os.path.basename(file_path)
        frame_id = base.split("distance_map_frame")[-1].split(".npy")[0]

        fig, ax = plt.subplots(figsize=(6, 6))

        im0 = ax.imshow(data, cmap='RdBu', vmin=0, vmax=180)

        ax.set_xlabel("Bin Index", fontsize=18)
        ax.set_ylabel("Bin Index", fontsize=18)
        ax.tick_params(axis='both', which='major', labelsize=18, width=2, length=6)

        cbar = fig.colorbar(im0, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Simulation Distance", fontsize=18)
        cbar.ax.tick_params(labelsize=18)

        output_path = os.path.join(
            traj_analysis_folder,
            f"frame{frame_id}_AE{AE}_B{B}.png"
        )

        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

        print(f"Saved: {output_path}")




# ----------------------------------------------------------
# Contact Map and Pairwise Distances with Standard Deviation
# ----------------------------------------------------------

def plot_linear_and_distance_maps_large(
    trajectory_analysis_folder,
    save_path=None
):
    
    """
    Plot ONLY:
    1) Linear contact map (avg lower / std upper)
    2) Pairwise distance map (avg lower / std upper)

    Uses precomputed grand-average .npy files.
    """

    apply_serif_font()


    contact_avg_path = os.path.join(
        trajectory_analysis_folder,
        "average_contact_map_all_runs.npy"
    )
    contact_std_path = os.path.join(
        trajectory_analysis_folder,
        "std_contact_map_all_runs.npy"
    )
    dist_avg_path = os.path.join(
        trajectory_analysis_folder,
        "average_distance_map_all_runs.npy"
    )
    dist_std_path = os.path.join(
        trajectory_analysis_folder,
        "std_distance_map_all_runs.npy"
    )

    if not all(os.path.exists(p) for p in [
        contact_avg_path, contact_std_path,
        dist_avg_path, dist_std_path
    ]):
        raise FileNotFoundError(
            "Missing one or more grand-average files in trajectory folder"
        )

    global_avg_contact = np.load(contact_avg_path)
    global_std_contact = np.load(contact_std_path)
    global_avg_distance = np.load(dist_avg_path)
    global_std_distance = np.load(dist_std_path)

    # Masks
    N = global_avg_contact.shape[0]
    lower_mask = np.tril(np.ones((N, N), dtype=bool))
    upper_mask = np.triu(np.ones((N, N), dtype=bool), k=1)


    # Plot
    colors_avg = [ "#053061", "#2166AC", "#92C5DE", "#F7F7F7", "#DA8683", "#DC6472", "#F70404"]

    custom_cbar_avg = mcolors.LinearSegmentedColormap.from_list("custom_blue", colors_avg)
    custom_cbar_std = "Purples"
    custom_cbar_avg_r = "RdBu"

    # Create Figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(34, 14))

    # Font- and tick-sizes
    fontsize_labels = 39
    tick_size = 42
    cbar_tick_size = 42


    # 1) Linear contact map
    im1_avg = ax1.imshow(
        np.where(lower_mask, global_avg_contact, np.nan),
        cmap=custom_cbar_avg,
        vmin=0,
        vmax=1
    )
    im1_std = ax1.imshow(
        np.where(upper_mask, global_std_contact, np.nan),
        cmap=custom_cbar_std,
        vmin=0,
        vmax=1
    )

    ax1.set_xlabel("Monomer Index", fontsize=fontsize_labels)
    ax1.set_ylabel("Monomer Index", fontsize=fontsize_labels)
    ax1.tick_params(axis='both', labelsize=tick_size, width=2, length=10)
    ax1.set_aspect('equal')

    divider1 = make_axes_locatable(ax1)
    cax1_avg = divider1.append_axes("bottom", size="6%", pad=1.3)
    cax1_std = divider1.append_axes("right", size="6%", pad=0.4)

    cbar1_avg = fig.colorbar(im1_avg, cax=cax1_avg, orientation="horizontal")
    cbar1_avg.set_label("Avg Contact", fontsize=fontsize_labels)
    cbar1_avg.ax.tick_params(labelsize=cbar_tick_size)

    cbar1_std = fig.colorbar(im1_std, cax=cax1_std, orientation="vertical")
    cbar1_std.set_label("STD Contact", fontsize=fontsize_labels)
    cbar1_std.ax.tick_params(labelsize=cbar_tick_size)



    # 2) Pairwise distance map
    vmax_dist = np.max(global_avg_distance)
    vmax_dist_std = np.max(global_std_distance)

    im2_avg = ax2.imshow(
        np.where(lower_mask, global_avg_distance, np.nan),
        cmap=custom_cbar_avg_r,
        vmin=0,
        vmax=150
    )
    im2_std = ax2.imshow(
        np.where(upper_mask, global_std_distance, np.nan),
        cmap=custom_cbar_std,
        vmin=0,
        vmax=50
    )

    ax2.set_xlabel("Monomer Index", fontsize=fontsize_labels)
    ax2.set_ylabel("Monomer Index", fontsize=fontsize_labels)
    ax2.tick_params(axis='both', labelsize=tick_size, width=2, length=10)
    ax2.set_aspect('equal')

    divider2 = make_axes_locatable(ax2)
    cax2_avg = divider2.append_axes("bottom", size="6%", pad=1.3)
    cax2_std = divider2.append_axes("right", size="6%", pad=0.4)

    cbar2_avg = fig.colorbar(im2_avg, cax=cax2_avg, orientation="horizontal")
    cbar2_avg.set_label("Avg Distance", fontsize=fontsize_labels)
    cbar2_avg.ax.tick_params(labelsize=cbar_tick_size)

    cbar2_std = fig.colorbar(im2_std, cax=cax2_std, orientation="vertical")
    cbar2_std.set_label("STD Distance", fontsize=fontsize_labels)
    cbar2_std.ax.tick_params(labelsize=cbar_tick_size)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
        plt.close(fig)
    else:
        plt.show()







# --------------------------------------------
# Polymer Configuration Initial and Last Frame
# --------------------------------------------


# This funtion is taken from the project thesis
def draw_sphere(ax, center, radius, alpha=0.12, color="gray"):
    u = np.linspace(0, 2*np.pi, 50)
    v = np.linspace(0, np.pi, 50)

    x = center[0] + radius * np.outer(np.cos(u), np.sin(v))
    y = center[1] + radius * np.outer(np.sin(u), np.sin(v))
    z = center[2] + radius * np.outer(np.ones_like(u), np.cos(v))

    ax.plot_surface(x, y, z, color=color, alpha=alpha, linewidth=0)



def plot_single_polymer_init_final(traj, seq_data, run_folder, lim,  polymer_index=0):

    # sequence (0/1 list)
    seq = np.array(seq_data)
    L = len(seq)

    # extract frames
    init = traj[0]
    final = traj[-1]

    frames = [init, final]
    names = ["initial", "final"]

    elev, azim = 20, 45  # fixed view for comparison

    for frame, name in zip(frames, names):

        # center
        com = np.mean(frame, axis=0)
        frame_centered = frame - com

        # radius of gyration
        Rg = radius_of_gyration(frame)

        # colors
        colors = np.where(seq == 0, "royalblue", "firebrick")

        fig = plt.figure(figsize=(18, 9))
        ax = fig.add_subplot(111, projection='3d')

        # bonds
        for i in range(L - 1):
            ax.plot(
                frame_centered[i:i+2, 0],
                frame_centered[i:i+2, 1],
                frame_centered[i:i+2, 2],
                color=colors[i],
                linewidth=2
            )

        # beads
        ax.scatter(
            frame_centered[:, 0],
            frame_centered[:, 1],
            frame_centered[:, 2],
            c=colors,
            s=60,
            edgecolors='black'
        )

        # Rg sphere
        draw_sphere(ax, com, Rg, alpha=0.12, color="gray")

        # styling (publication style)
        ax.set_xlabel("X", fontsize=26, labelpad=12)
        ax.set_ylabel("Y", fontsize=26, labelpad=12)
        ax.set_zlabel("Z", fontsize=26, labelpad=12)

        ax.set_xlim(-lim,lim )
        ax.set_ylim(-lim, lim)
        ax.set_zlim(-lim, lim)

        ax.tick_params(axis='both', which='major', labelsize=20, length=10, width=2)
        ax.tick_params(axis='z', labelsize=20, length=10, width=2)

        ax.xaxis.set_major_locator(mticker.MaxNLocator(5))
        ax.yaxis.set_major_locator(mticker.MaxNLocator(5))
        ax.zaxis.set_major_locator(mticker.MaxNLocator(5))

        ax.set_box_aspect([1, 1, 1])
        ax.view_init(elev=elev, azim=azim)

        plt.tight_layout()

        # SAVE
        save_path = os.path.join(run_folder, f"polymer_{name}.png")
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"Saved: {save_path}")







# -------------------------------------------
# Distance Scaling Functions for Simulations
# -------------------------------------------




def plot_single_run_with_std(
    csv_path,
    outpath,
    label="Run",
    color="C0",
    y_scale=1.0,
    max_gdist=120_000
):
    """
    Plot a single run's genomic distance curve with mean ± std.

    Parameters
    ----------
    csv_path : str
        Path to df_stats.csv for one run
    outpath : str
        Where to save the figure
    label : str
        Legend label
    color : str
        Line color
    y_scale : float
        Scaling factor for y-axis (optional)
    max_gdist : float
        Upper cutoff for genomic distance
    """

    df = pd.read_csv(csv_path)


    # Clean + filter
    df = df.copy()
    df["g_dist"] = pd.to_numeric(df["g_dist"])
    df["dist_mean"] = pd.to_numeric(df["dist_mean"])
    df["dist_std"] = pd.to_numeric(df["dist_std"])

    df = df[df["g_dist"] <= max_gdist]
    df = df.sort_values("g_dist")


    # Extract curves
    x = df["g_dist"]
    y = df["dist_mean"] * y_scale
    std = df["dist_std"] * y_scale


    # Plot
    plt.figure(figsize=(6, 4))

    plt.plot(
        x, y,
        linewidth=3,
        color=color,
        label=label
    )

    plt.fill_between(
        x,
        y - std,
        y + std,
        color=color,
        alpha=0.25
    )

    plt.xlabel("Genomic distance (kb)", fontsize=14)
    plt.ylabel("Spatial distance", fontsize=14)

    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    plt.legend(fontsize=12)

    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()







def plot_exp_vs_sim(
    exp_df,
    sim_csv_paths,
    outpath,
    labels,
    colors,
    y_scale=1.0
):

    # Load simulation CSVs
    sim_dfs = [pd.read_csv(path) for path in sim_csv_paths]


    # Filter experiment
    exp_df = exp_df.copy()
    exp_df = exp_df[(exp_df["g_dist"] >= 0) & (exp_df["g_dist"] <= 120_000)]
    exp_df = exp_df.sort_values("g_dist")


    # Filter simulations
    cleaned_sims = []
    for df in sim_dfs:
        df = df.copy()
        df = df[(df["g_dist"] >= 0) & (df["g_dist"] <= 120_000)]
        df = df.sort_values("g_dist")
        cleaned_sims.append(df)

    # -------------------------
    # Plot
    # -------------------------
    plt.figure(figsize=(6, 4))

    # Experiment (median curve)
    sns.lineplot(
        data=exp_df,
        x="g_dist",
        y="dist",
        estimator=np.median,
        errorbar=None,
        linewidth=3,
        label="Experiment",
        color="#257749"
    )

    # Simulations (already pre-averaged)
    for sim_df, label, color in zip(cleaned_sims, labels, colors):

        plt.plot(
            sim_df["g_dist"],
            sim_df["dist_mean"] * y_scale,
            linewidth=3,
            label=label,
            color=color
        )


    # Labels
    plt.xlabel("Genomic distance (kb)", fontsize=14)
    plt.ylabel("Spatial distance", fontsize=14)

    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    plt.legend(fontsize=12)

    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()





