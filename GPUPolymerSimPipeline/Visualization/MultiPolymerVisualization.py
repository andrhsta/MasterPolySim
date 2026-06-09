"""
----------------------------------------------------------
Polymer Visualization Functions for Multi-polymer Systems
----------------------------------------------------------

"""

# Importing modules
import os
import sys
import json
import numpy as np
import pandas as pd
import seaborn as sns


import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from skimage.transform import resize
from mpl_toolkits.axes_grid1 import make_axes_locatable


# For IDUN: Loading path to GPUPolymerSimPipeline and polychrom
sys.path.append('/cluster/work/andrhsta/Masteroppgave')

# For importing LoopTrace when using IDUN:
sys.path.append("/cluster/work/andrhsta/Masteroppgave/TracingData")
sys.path.append("/cluster/work/andrhsta/Masteroppgave/TracingData/TraceAnalysis")

from looptrace import trace_analysis_functions as tr # Custom functions for analyzing tracing data



# Importing polymer sequence from sequence module
from GPUPolymerSimPipeline.Analysis.PolymerAnalysis import (radius_of_gyration
                                                            )

from GPUPolymerSimPipeline.Visualization.SinglePolymerVisualization import (apply_serif_font)
from GPUPolymerSimPipeline.ExperimentalData.ExperimentalData import (replace_nan_with_diagonal_median, calc_distances, round_to_nearest_multiple)


apply_serif_font() 

# -----------------------------------
# Splitting Trajectory into Polymers
# ----------------------------------

def split_polymers(coords, n_polymers, L):
    """
    Returns list of trajectories:
    [(frames, L, 3), (frames, L, 3), ...]
    """
    polymers = []

    for p in range(n_polymers):
        start = p * L
        end = (p + 1) * L
        polymers.append(coords[:, start:end, :])

    return polymers


# ---------------------------------
# Multi-Polymer Frame Visualization
# ---------------------------------

def get_polymer_colors(p):
    POLYMER_COLORS = [
        ("green",   "#2ca02c", "#98df8a"),
        ("blue",    "#1f77b4", "#aec7e8"),
        ("red",     "#d62728", "#ff9896"),
        ("purple",  "#9467bd", "#c5b0d5"),
        ("orange",  "#ff7f0e", "#ffbb78"),
        ("brown",   "#8c564b", "#c49c94"),
        ("cyan",    "#17becf", "#9edae5"),
    ]
    _, dark, light = POLYMER_COLORS[p % len(POLYMER_COLORS)]
    return light, dark


def draw_sphere(ax, radius, alpha=0.1, color="gray"):
    u = np.linspace(0, 2*np.pi, 50)
    v = np.linspace(0, np.pi, 50)

    x = radius * np.outer(np.cos(u), np.sin(v))
    y = radius * np.outer(np.sin(u), np.sin(v))
    z = radius * np.outer(np.ones_like(u), np.cos(v))

    ax.plot_surface(x, y, z, color=color, alpha=alpha, linewidth=0)







def plot_frame(positions, full_sequence, n_polymers, L, lim,  save_path):

    print("\n--- ENTER plot_frame ---")
    print("Saving to:", save_path)

    positions = positions - np.mean(positions, axis=0)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    draw_sphere(ax, R, alpha=0.08, color="gray")

    for p in range(n_polymers):
        A_color, B_color = get_polymer_colors(p)

        start = p * L
        end = (p + 1) * L

        chain = positions[start:end]
        seq = full_sequence[start:end]

        for i in range(len(chain) - 1):
            color = A_color if seq[i] == 0 else B_color
            ax.plot(
                chain[i:i+2, 0],
                chain[i:i+2, 1],
                chain[i:i+2, 2],
                color=color,
                linewidth=2
            )

        bead_colors = [A_color if s == 0 else B_color for s in seq]

        ax.scatter(
            chain[:, 0],
            chain[:, 1],
            chain[:, 2],
            c=bead_colors,
            s=40,
            edgecolors='black'
        )

    
    ax.set_xlabel("X", fontsize=15, labelpad=15)
    ax.set_ylabel("Y", fontsize=15, labelpad=15)
    ax.set_zlabel("Z", fontsize=15, labelpad=15)

    ax.set_xlim(-lim,lim )
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-lim, lim)

    ax.tick_params(axis='both', which='major', labelsize=15, length=40, width=3)
    ax.tick_params(axis='z', labelsize=15, length=40, width=3)
    import matplotlib.ticker as mticker
    ax.xaxis.set_major_locator(mticker.MaxNLocator(5))
    ax.yaxis.set_major_locator(mticker.MaxNLocator(5))
    ax.zaxis.set_major_locator(mticker.MaxNLocator(5))
   
    ax.view_init(20, 45)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print("Figure saved")

    plt.close() 








def plot_first_and_last(run_folder):

    print("\nStarting plot_first_and_last")

    traj = np.load(os.path.join(run_folder, "trajectory.npy"))

    with open(os.path.join(run_folder, "sequence.json")) as f:
        seq_data = json.load(f)

    with open(os.path.join(run_folder, "params.json")) as f:
        params = json.load(f)

    R = params["confinement_radius"]

    n_polymers = seq_data["n_polymers"]
    L = seq_data["L"]

    full_sequence = []
    for p in seq_data["polymers"]:
        full_sequence.extend(p["sequence"])
    full_sequence = np.array(full_sequence)

    first_frame = traj[0]
    last_frame = traj[-1]

    first_path = os.path.join(run_folder, "first_frame.png")
    last_path = os.path.join(run_folder, "last_frame.png")

    print("Saving first frame to:", first_path)
    plot_frame(first_frame, full_sequence, n_polymers, L,  150, first_path)

    print("Saving last frame to:", last_path)
    plot_frame(last_frame, full_sequence, n_polymers, L, 150, last_path)









# -----------------------------------------------
# Radius of Gyration - per polymer and per system
# -----------------------------------------------


def load_trajectory(path):
    traj = np.load(path)
    print("Trajectory shape:", traj.shape)
    return traj


def load_sequence_info(sequence_path):
    with open(sequence_path, "r") as f:
        data = json.load(f)

    n_polymers = data["n_polymers"]
    L = data["L"]

    print(f"Polymers: {n_polymers}, L: {L}")
    return n_polymers, L



def plot_rg_per_polymer(traj_path, sequence_path, save_interval=1000):

    coords = load_trajectory(traj_path)
    n_polymers, L = load_sequence_info(sequence_path)

    polymers = split_polymers(coords, n_polymers, L)
    polymer_colors = ["#2ca02c", "#1f77b4", "#d62728", "#9467bd", "#ff7f0e", "#8c564b", "#17becf"]

    # POLYMER_COLORS = [
    # ("green",   "#2ca02c", "#98df8a"),
    # ("blue",    "#1f77b4", "#aec7e8"),
    # ("red",     "#d62728", "#ff9896"),
    # ("purple",  "#9467bd", "#c5b0d5"),
    # ("orange",  "#ff7f0e", "#ffbb78"),
    # ("brown",   "#8c564b", "#c49c94"),
    # ("cyan",    "#17becf", "#9edae5"),



    plt.figure(figsize=(14, 8))
    ax = plt.gca()

    for p, polymer_coords in enumerate(polymers):

        rg_values = radius_of_gyration(polymer_coords)
        time = np.arange(len(rg_values)) * save_interval

        color = polymer_colors[p % len(polymer_colors)]

        ax.plot(
            time,
            rg_values,
            lw=2,
            color=color,
            label=f"Polymer {p}"
        )


    # Styling

    ax.set_ylim(0, 100)
    ax.set_xlabel("Simulation step", fontsize=27)
    ax.set_ylabel(r"$R_g$", fontsize=28)

    ax.tick_params(axis='both', labelsize=27)
    ax.grid(True, linestyle='--', alpha=0.5)

    ax.legend(fontsize=18)

    # Save in same folder
    output_dir = os.path.dirname(traj_path)
    save_path = os.path.join(output_dir, "Rg_per_polymer_1.png")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved Rg plot → {save_path}")









def plot_rg_system_multiple(base_paths, save_interval=1000):


    plt.figure(figsize=(14, 8))
    ax = plt.gca()
    colors = ["#6F9796", "#E9C46A", "#BA4F34"]
    labels = [
                "No confinement",
                r"$\rho = 0.1$ ",
                r"$\rho = 0.01$ "
            ]


    for i, base_path in enumerate(base_paths):

        print(f"\nProcessing: {base_path}")

        traj_path = f"{base_path}/trajectory.npy"
        seq_path  = f"{base_path}/sequence.json"

        coords = load_trajectory(traj_path)
        n_polymers, L = load_sequence_info(seq_path)

        # Compute system Rg
        rg_values = radius_of_gyration(coords)
        time = np.arange(len(rg_values)) * save_interval

        # Use folder name as label
        label = labels[i]
        ax.plot(time, rg_values, lw=3, color=colors[i % len(colors)], label=label)

    # Styling
    ax.set_xlabel("Simulation step", fontsize=27)
    ax.set_ylabel(r"$R_g$", fontsize=27)

    ax.tick_params(axis='both', labelsize=28)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(fontsize=27)

    # Optional y-axis limits
    # ax.set_ylim(0, 50)

    # Save
    save_path = "Rg_system_comparison.png"
    plt.tight_layout()
    plt.savefig(
        save_path,
        dpi=300,
        bbox_inches='tight'
    )

    plt.close()

    print(f"\nSaved comparison plot → {save_path}")









# -----------------------------------------
# Intra- and Interchromosomal Contact Maps
# -----------------------------------------


# Plot Intra- and Interchromsomal Maps
def plot_contact_map(
    traj_path,
    cutoff=40.0,
    k=5
):

    run_folder = os.path.dirname(traj_path)
    system_folder = os.path.dirname(run_folder)

    analysis_dir = os.path.join(system_folder, "Analysis")

    map_path = os.path.join(analysis_dir, "contact_map.npz")

    data = np.load(map_path)

    full_map = data["full_map"]
    n_polymers = int(data["n_polymers"])
    L = int(data["L"])

    print(f"Loaded contact map from: {map_path}")

    # Figure
    fig, ax = plt.subplots(figsize=(14, 10))
    im = ax.imshow(full_map, cmap="magma", origin="upper")

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.035, pad=0.02, shrink=1)
    cbar.set_label("Contact probability", fontsize=26)
    cbar.ax.tick_params(labelsize=22)

    # Labels
    ax.set_xlabel("Bead index", fontsize=28)
    ax.set_ylabel("Bead index", fontsize=28)

    ax.tick_params(axis='both', which='major', labelsize=22)

    # Polymer boundaries
    boundaries = [i * L for i in range(1, n_polymers)]
    for b in boundaries:
        ax.axhline(b - 0.5, color="white", linewidth=1.0, alpha=0.8)
        ax.axvline(b - 0.5, color="white", linewidth=1.0, alpha=0.8)

    # Rectangles
    for i in range(n_polymers):
        for j in range(n_polymers):
            rect = patches.Rectangle(
                (j * L, i * L),
                L,
                L,
                linewidth=0.5,
                edgecolor="white",
                facecolor="none",
                alpha=0.3
            )
            ax.add_patch(rect)

    # Save
    save_path = os.path.join(
    analysis_dir,
    f"contact_map_stride{k}_cutoff{cutoff}.png"
)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved: {save_path}")











                                                                            
# ------------------------------------------------------------------
# Chromatin Tracing Data vs Simulated Averaged Pairwise Distance Map
# ------------------------------------------------------------------


# Average over each polymer
def plot_simulation_vs_tracing_half_polymeravg(
    traj_files,
    output_folder_name="polymer_halfplots_vmax=max_tracing",
    polymer_file_name="polymer_distance_maps.npz"):


    # Load DNA-tracing data
    traces = pd.read_csv(
        "/cluster/work/andrhsta/Masteroppgave/Examples/GPUPolymerSimPipeline/ExperimentalData/combined_dataset_v2.csv.gz",
        low_memory=False
    )

    traces['tmp'] = traces['exp'] + '_' + traces['region'] + '_' + traces['trace_id'].astype(str)
    trace_id_map = {tmp:i for i,tmp in enumerate(traces.tmp.unique())}
    traces['trace_id'] = traces['tmp'].map(trace_id_map)

    region = 'Chr2_full'
    cell_type = 'IMR90_ctrl'

    df = traces.loc[(traces.region==region) & (traces.cell_type==cell_type) & (traces.g_pos>0)]

    full_count = 239 if region == 'Chr2_full' else 84
    valid_trace_ids = df.groupby('trace_id').count()['fit_id'] == full_count
    df = df[df.trace_id.isin(valid_trace_ids[valid_trace_ids].index)]

    df = tr.tracing_length_qc(df, min_length=30)
    pwds = tr.pwd_calc(df)
    obs_dists = np.nanmean(pwds, axis=0)
    obs_dists = replace_nan_with_diagonal_median(obs_dists)

    for traj_file in traj_files:

        run_folder = os.path.dirname(traj_file)
        analysis_dir = os.path.join(run_folder, "Analysis")

        # Load metadata
        with open(os.path.join(run_folder, "sequence.json"), "r") as f:
            seq_data = json.load(f)

        n_polymers = seq_data["n_polymers"]

        print(f"\nProcessing: {traj_file}")
        print(f"Polymers: {n_polymers}")

        # Load precomputed polymer distance maps
        data = np.load(os.path.join(analysis_dir, polymer_file_name))
        polymer_maps = data["polymer_maps"]

        run_name = os.path.basename(run_folder)

        # Save output inside run 
        output_dir = os.path.join(analysis_dir, output_folder_name)
        os.makedirs(output_dir, exist_ok=True)

        # Loop over polymers (no recomputation)
        for p in range(n_polymers):

            print(f"Polymer {p}")

            sim_avg = polymer_maps[p]

            # Resize tracing matrix
            tracing_resized = resize(
                obs_dists,
                sim_avg.shape,
                order=1,
                mode='reflect',
                anti_aliasing=True,
                preserve_range=True
            )

            # Create upper/lower masks
            N = sim_avg.shape[0]

            upper_mask = np.triu(np.ones((N, N), dtype=bool), k=1)
            lower_mask = np.tril(np.ones((N, N), dtype=bool), k=-1)

            upper_triangle = np.where(upper_mask, sim_avg, np.nan)
            lower_triangle = np.where(lower_mask, tracing_resized, np.nan)

            # Create plot
            fig, ax = plt.subplots(figsize=(14, 14))

            im_sim = ax.imshow(upper_triangle, cmap='RdBu', vmin=0, vmax=np.nanmax(upper_triangle))
            im_tracing = ax.imshow(lower_triangle, cmap='RdBu',  vmin=200, vmax=3500)

            # Labels
            ax.set_xlabel("Monomer Index", fontsize=28, fontweight='bold')
            ax.set_ylabel("Monomer Index", fontsize=28, fontweight='bold')

            ax.tick_params(axis='both', which='major', labelsize=30)
            ax.tick_params(axis='both', which='minor', labelsize=30)

            # Simulation colorbar
            cbar_sim = fig.colorbar(im_sim, ax=ax, fraction=0.046, pad=0.04)
            cbar_sim.set_label("Simulation Distance (nm)", fontsize=30, fontweight='bold')
            cbar_sim.ax.tick_params(labelsize=30)

            # Tracing colorbar
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("bottom", size="5%", pad=0.9)
            cbar_tracing = plt.colorbar(im_tracing, cax=cax, orientation='horizontal')
            cbar_tracing.set_label("DNA-Tracing Distance (nm)", fontsize=30, fontweight='bold')
            cbar_tracing.ax.tick_params(labelsize=30)

            save_path = os.path.join(
                output_dir,
                f"{run_name}_polymer_{p}_halfplot.png"
            )

            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            plt.close(fig)

            print(f"    saved -> polymer {p}")






# Average of all polymers                                                    
def plot_sim_vs_tracing_half_map_allpolymeravg(
    run_folder,
    vmin_sim=0,
    vmax_sim=200,
    vmin_tracing=200,
    vmax_tracing=3500,
    AE=0.0,
    B=0.32,
    save=True,
    global_file_name="global_distance_map.npz"
    ):

    # Load precomputed global simulation map
    run_folder = run_folder
    system_folder = os.path.dirname(run_folder)
    analysis_dir = os.path.join(system_folder, "Analysis")

    global_path = os.path.join(analysis_dir, global_file_name)

    if not os.path.isfile(global_path):
        raise FileNotFoundError(f"No global map found in {analysis_dir}")

    data = np.load(global_path)
    sim_avg = data["global_avg"]

    print("Loaded global simulation map:", sim_avg.shape)


    # Load DNA-tracing data
    traces = pd.read_csv(
        "/cluster/work/andrhsta/Masteroppgave/Examples/GPUPolymerSimPipeline/ExperimentalData/combined_dataset_v2.csv.gz",
        low_memory=False
    )

    traces['tmp'] = traces['exp'] + '_' + traces['region'] + '_' + traces['trace_id'].astype(str)
    trace_id_map = {tmp:i for i,tmp in enumerate(traces.tmp.unique())}
    traces['trace_id'] = traces['tmp'].map(trace_id_map)

    region = 'Chr2_full'
    cell_type = 'IMR90_ctrl'

    df = traces.loc[(traces.region==region) & (traces.cell_type==cell_type) & (traces.g_pos>0)]

    full_count = 239 if region == 'Chr2_full' else 84
    valid_trace_ids = df.groupby('trace_id').count()['fit_id'] == full_count
    df = df[df.trace_id.isin(valid_trace_ids[valid_trace_ids].index)]

    df = tr.tracing_length_qc(df, min_length=30)
    pwds = tr.pwd_calc(df)
    obs_dists = np.nanmean(pwds, axis=0)
    obs_dists = replace_nan_with_diagonal_median(obs_dists)


    # Resize tracing data
    tracing_resized = resize(
        obs_dists,
        sim_avg.shape,
        order=1,
        mode='reflect',
        anti_aliasing=True,
        preserve_range=True
    )

    # Build half matrices
    N = sim_avg.shape[0]

    upper_mask = np.triu(np.ones((N, N), dtype=bool), k=1)
    lower_mask = np.tril(np.ones((N, N), dtype=bool), k=-1)

    upper_triangle = np.where(upper_mask, sim_avg + 1e-5, np.nan)
    lower_triangle = np.where(lower_mask, tracing_resized + 1e-5, np.nan)

    # Plot
    fig, ax = plt.subplots(figsize=(15, 24))

    im_sim = ax.imshow(upper_triangle, cmap="RdBu", vmin=vmin_sim, vmax=vmax_sim)
    im_tracing = ax.imshow(lower_triangle, cmap="RdBu", vmin=vmin_tracing, vmax=vmax_tracing)

    ax.set_xlabel("Monomer Index", fontsize=30)
    ax.set_ylabel("Monomer Index", fontsize=30)

    ax.tick_params(axis="both", which="major", labelsize=35)

    # Colorbars
    divider = make_axes_locatable(ax)

    cax_sim = divider.append_axes("right", size="4%", pad=0.15)
    cbar_sim = plt.colorbar(im_sim, cax=cax_sim)
    cbar_sim.set_label("Simulation Distance (nm)", fontsize=35)
    cbar_sim.ax.tick_params(labelsize=30)

    cax_tr = divider.append_axes("bottom", size="4%", pad=1)
    cbar_tr = plt.colorbar(im_tracing, cax=cax_tr, orientation="horizontal")
    cbar_tr.set_label("DNA-Tracing Distance (nm)", fontsize=35)
    cbar_tr.ax.tick_params(labelsize=35)

    # Save
    output_path = os.path.join(
                            analysis_dir,
                            f"sim_vs_tracing_half_AE{AE}_B{B}.png"
                        )

    if save:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")

    plt.close(fig)

    print(f"Saved figure to {output_path}")

    return sim_avg










# ---------------------------------------
# Distance Scaling Curve for Simulations
# ---------------------------------------



def plot_exp_vs_sim(
    exp_df,
    sim_csv_files,
    labels,
    outpath,
    x_scale=1.0,
    y_scale=1.0,
    x_cut=120_000
):

    exp_df = exp_df[exp_df["g_dist"] <= x_cut].copy()

    plt.figure(figsize=(6, 4))

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

    colors = ["#E9C46A", "#87438F", "#57C3AD", "#7288B3"]

    for idx, file_path in enumerate(sim_csv_files):

        # ---- FIX: accept DF or path ----
        if isinstance(file_path, str):
            df = pd.read_csv(file_path)
        else:
            df = file_path.copy()

        # ---- standardize column ----
        if "dist_mean" in df.columns:
            df = df.rename(columns={"dist_mean": "dist"})

        df = df[df["g_dist"] <= x_cut].sort_values("g_dist")

        plt.plot(
            df["g_dist"] * x_scale,
            df["dist"] * y_scale,
            linewidth=2.5,
            color=colors[idx % len(colors)],
            label=labels[idx]
        )

    plt.xlabel("Genomic distance (kb)", fontsize=14)
    plt.ylabel("Spatial distance", fontsize=14)
    plt.legend()
    plt.tight_layout()

    plt.savefig(outpath, dpi=300)
    plt.close()






def plot_exp_single_multi_scaled(
    exp_df,
    single_csv,
    multi_csv,
    outpath,
    single_scale=1.0,
    multi_scale=1.0,
    x_cut=120_000
):

    exp_df = exp_df[(exp_df["g_dist"] >= 0) & (exp_df["g_dist"] <= x_cut)]

    exp_curve = (
        exp_df.groupby("g_dist", as_index=False)["dist"]
        .median()
        .sort_values("g_dist")
    )

    single_df = pd.read_csv(single_csv)
    multi_df = pd.read_csv(multi_csv)

    if "dist_mean" in single_df.columns:
        single_df = single_df.rename(columns={"dist_mean": "dist"})
    if "dist_mean" in multi_df.columns:
        multi_df = multi_df.rename(columns={"dist_mean": "dist"})

    single_df = single_df[(single_df["g_dist"] >= 0) & (single_df["g_dist"] <= x_cut)]
    multi_df = multi_df[(multi_df["g_dist"] >= 0) & (multi_df["g_dist"] <= x_cut)]

    plt.figure(figsize=(6, 4))

    sns.lineplot(
        data=exp_curve,
        x="g_dist",
        y="dist",
        linewidth=3,
        color="#257749",
        label="Experiment"
    )

    plt.plot(
        single_df["g_dist"],
        single_df["dist"] * single_scale,
        linewidth=3,
        color="#49569C",
        label="Single polymer system"
    )

    plt.plot(
        multi_df["g_dist"],
        multi_df["dist"] * multi_scale,
        linewidth=3,
        color="#82277E",
        label="Multi-polymer system"
    )

    plt.xlabel("Genomic distance (kb)")
    plt.ylabel("Spatial distance (nm)")
    plt.legend()

    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()











# ---------------------------------------
# Boxplot for Monitoring Polymer Overlap
# ---------------------------------------

def plot_polymer_overlap_boxplot_from_npy(
    systems,
    label_dict,
    input_path,
    reference_line=1.0
):
    """
    Loads .npy results and creates boxplot.
    System labels (Ae, B, rho) are provided via label_dict.
    """



    # Folder for saving plots
    first_path = next(iter(systems.values()))
    run_folder = os.path.dirname(first_path)
    system_folder = os.path.dirname(run_folder)
    analysis_dir = os.path.join(system_folder, "Analysis")
    os.makedirs(analysis_dir, exist_ok=True)


    # Load data
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"Could not find: {input_path}")

    all_data = np.load(input_path, allow_pickle=False)

    # Ensure consistent ordering
    system_names = sorted(np.unique(all_data["system"]))


    # Pair colors 
    unique_pairs = sorted(np.unique(all_data["pair"]).astype(str))
    cmap = plt.get_cmap("tab20")

    pair_colors = {
        pair: cmap(i % 20)
        for i, pair in enumerate(unique_pairs)
    }


    # Boxplot data

    boxplot_data = [
        all_data["ratio"][all_data["system"] == s]
        for s in system_names
    ]

    # Labels
    xticklabels = []

    for s in system_names:

        if s in label_dict:
            pars = label_dict[s]
            label = (
                rf"$A_e={pars['Ae']}$"
                "\n"
                rf"$B={pars['B']}$"
                "\n"
                rf"$\rho={pars['rho']}$"
            )
        else:
            label = s

        xticklabels.append(label)


    # Plot
    fig, ax = plt.subplots(figsize=(15, 9))

    bp = ax.boxplot(
        boxplot_data,
        positions=np.arange(len(system_names)),
        widths=0.5,
        patch_artist=True,
        showfliers=False
    )

    for box in bp["boxes"]:
        box.set(facecolor="lightgray", alpha=0.5, linewidth=2)

    for whisker in bp["whiskers"]:
        whisker.set(linewidth=2)

    for cap in bp["caps"]:
        cap.set(linewidth=2)

    for median in bp["medians"]:
        median.set(linewidth=2, color="black")


    # Scatter overlay 
    rng = np.random.default_rng(0)

    for idx, s in enumerate(system_names):

        mask = all_data["system"] == s
        rows = all_data[mask]

        for row in rows:

            pair = str(row["pair"])

            ax.scatter(
                idx + rng.uniform(-0.08, 0.08),
                row["ratio"],
                color=pair_colors.get(pair, "black"),
                edgecolors="black",
                s=80,
                alpha=0.85
            )


    # Reference line
    ax.axhline(
        reference_line,
        color="lime",
        linestyle="--",
        linewidth=2
    )


    # Labels
    ax.set_xticks(np.arange(len(system_names)))
    ax.set_xticklabels(xticklabels)

    ax.set_ylabel(r"$d/(R_{g,i}+R_{g,j})$", fontsize=18)
    ax.set_xlabel("System parameters", fontsize=18)

    ax.tick_params(axis="both", labelsize=14)


    # Legend 
    added = set()

    for pair in unique_pairs:
        if pair not in added:
            ax.scatter(
                [],
                [],
                color=pair_colors[pair],
                label=pair,
                s=80,
                edgecolors="black"
            )
            added.add(pair)

    ax.legend(
        title="Polymer pairs",
        bbox_to_anchor=(1.05, 1),
        loc="upper left"
    )


    # Save
    plt.tight_layout()

    save_path = os.path.join(analysis_dir, "polymer_overlap_boxplot.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    print("Saved:", save_path)