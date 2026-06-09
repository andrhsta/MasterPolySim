"""
---------------------------------------------------------------
Functions for Analysis and Visualization of Polymer Simulations
---------------------------------------------------------------

Functions for analysing polymer simulation data, including radius of gyration,
contacts between monomers and pairwise distances. 

"""

# Importing modules
import os
import sys
import json
import itertools
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")


sys.path.append("/cluster/work/andrhsta/Masteroppgave/TracingData")
sys.path.append("/cluster/work/andrhsta/Masteroppgave/TracingData/TraceAnalysis")

from looptrace import trace_analysis_functions as tr
from scipy.spatial.distance import pdist, squareform 



# ---------------------------------------------------------------------------------------------------------------------------------------------
# Single-Polymer Simulations
# ----------------------------------------------------------------------------------------------------------------------------------------------



# -----------------------------------------------
# Calculating Radius of Gyration from Trajectory
# -----------------------------------------------

def radius_of_gyration(positions, trajectory_path=None):
    """
    Calculate the Radius of Gyration for a set of positions from single or multiple frames.
    
    Parameters:
        positions (numpy.ndarray): Array of positions
            - single frame: shape (N, 3)
            - multiple frames: shape (num_frames, N, 3)
    
    Returns:
        float or numpy.ndarray w. Radius of Gyration
            - single frame: float
            - multiple frames: numpy array of shape (num_frames,)
    """

    if positions.ndim == 2:
        # Single frame
        center_of_mass = np.mean(positions, axis=0)
        rg_squared = np.mean(np.sum((positions - center_of_mass) ** 2, axis=1))
        result = np.sqrt(rg_squared)
    else:
        # Multiple frames
        center_of_mass = np.mean(positions, axis=1, keepdims=True)  # shape (n_frames, 1, n_dim)
        rg_squared = np.mean(np.sum((positions - center_of_mass)**2, axis=2), axis=1)  # shape (n_frames,)
        result = np.sqrt(rg_squared)

        if trajectory_path is not None:
            folder = os.path.join(os.path.dirname(trajectory_path), "Analysis")
            os.makedirs(folder, exist_ok=True)

            filename = os.path.join(folder, "rg.npy")
            np.save(filename, result)



    return result
    


# ----------------------------------
# Functions for Contact Map Analysis
# ----------------------------------

def compute_contact_map(positions, cutoff=6.0):
    """
    Compute contact map from positions.
    
    parameters:
        positions: np.ndarray of shape (N, 3) representing 3D coordinates of N monomers.
        cutoff: float, distance threshold to define a contact.

    returns:
        contact_map: np.ndarray of shape (N, N), binary contact map showing contacts between monomers.
    """

    N = len(positions)
    contact_map = np.zeros((N, N))
    for i in range(N):
        for j in range(i + 1, N):
            distance = np.linalg.norm(positions[i] - positions[j])
            if distance < cutoff:
                contact_map[i, j] = 1
                contact_map[j, i] = 1


    return contact_map






def compute_average_contact_map(coords_over_time, trajectory_path, cutoff = 6.0):
    """
    Compute average and standard deviation contact maps for a selection of frames.

    parameters:
        coords_over_time: np.ndarray of shape (num_frames, N, 3)
            All coordinates over time.
        frame_indices: list of int
            Indices of frames to include in the average.
        cutoff: float
            Distance threshold to define contacts.

    returns:
        avg_contact_map: np.ndarray of shape (N, N)
            Averaged contact map over the selected frames.
        std_contact_map: np.ndarray of shape (N, N)
            Standard deviation of contact maps over the selected frames.
    """

    print("\n[DEBUG] compute_average_contact_map ENTERED")
    print("[DEBUG] coords_over_time type:", type(coords_over_time))
    print("[DEBUG] coords_over_time is None:", coords_over_time is None)

    if coords_over_time is None:
        raise ValueError("coords_over_time is None → upstream failure")
        
    contact_maps = []

    for frame in coords_over_time:
        contact_maps.append(compute_contact_map(frame, cutoff))

    contact_maps = np.array(contact_maps)

    avg_contact_map = np.mean(contact_maps, axis=0)
    std_contact_map = np.std(contact_maps, axis=0)


    folder = os.path.join(trajectory_path, "Analysis")
    os.makedirs(folder, exist_ok=True)

    avg_path = os.path.join(folder, "average_contact_map.npy")
    std_path = os.path.join(folder, "std_contact_map.npy")

    np.save(avg_path, avg_contact_map)
    np.save(std_path, std_contact_map)

    return avg_contact_map, std_contact_map







def average_run_contact_maps(trajectory_folder):
    """
    Average all average_contact_map.npy files found in run folders
    inside a trajectory folder.

    Parameters
    ----------
    trajectory_folder : str
        Path to the trajectory folder containing run_* directories.

    Returns
    -------
    grand_avg_map : ndarray
        Mean contact map across runs.
    grand_std_map : ndarray
        Standard deviation across runs.
    """

    maps = []

    # Find all run folders
    run_folders = sorted(
        d for d in os.listdir(trajectory_folder)
        if os.path.isdir(os.path.join(trajectory_folder, d))
        and d.startswith("run_")
    )

    if len(run_folders) == 0:
        raise ValueError(f"No run folders found in {trajectory_folder}")

    for run_folder in run_folders:

        map_path = os.path.join(
                    trajectory_folder,
                    run_folder,
                    "Analysis",
                    "average_contact_map.npy"
                )
        
        if os.path.exists(map_path):
            maps.append(np.load(map_path))
        else:
            print(f"Warning: {map_path} not found")

    if len(maps) == 0:
        raise ValueError("No average_contact_map.npy files found")

    maps = np.array(maps)

    grand_avg_map = np.mean(maps, axis=0)
    grand_std_map = np.std(maps, axis=0)

    folder = os.path.join(trajectory_folder, "Analysis")
    os.makedirs(folder, exist_ok=True)

    np.save(
        os.path.join(folder, "average_contact_map_all_runs.npy"),
        grand_avg_map
    )

    np.save(
        os.path.join(folder, "std_contact_map_all_runs.npy"),
        grand_std_map
    )

    return grand_avg_map, grand_std_map




# ----------------------------------------
# Functions for Pairwise Distance Analysis
# ----------------------------------------

def compute_distance_map(positions, trajectory_path=None, save_path = "distance_map.npy"):
    """
    Compute the full pairwise Euclidean distance matrix 
    for a single frame of coordinates.
    
    Parameters
    ----------
    positions : ndarray of shape (N, 3)
        Cartesian coordinates of N monomers.

    Returns
    -------
    dist_map : ndarray of shape (N, N)
        Distance matrix where entry (i, j) is the distance
        between monomer i and monomer j.
    """

    diff = positions[:, None, :] - positions[None, :, :]
    dist_map = np.linalg.norm(diff, axis=2)

    if trajectory_path is not None:
        folder = os.path.join(trajectory_path, "Analysis")
        os.makedirs(folder, exist_ok=True)
        dist_path = os.path.join(folder, save_path)
        np.save(dist_path, dist_map)

    return dist_map





def compute_average_distance_map(coords_over_time, trajectory_path):
    """
    Compute the average and standard deviation of pairwise 
    distance matrices over selected frames.

    Parameters
    ----------
    coords_over_time : ndarray of shape (num_frames, N, 3)
        All coordinates from the simulation.
    frame_indices : list of int
        Frame indices to include in averaging.

    Returns
    -------
    avg_distance_map : ndarray of shape (N, N)
    std_distance_map : ndarray of shape (N, N)
    """
    
    dist_maps = []

    for frame in coords_over_time:
        dist_maps.append(compute_distance_map(frame))

    dist_maps = np.array(dist_maps)

    avg_distance_map = np.mean(dist_maps, axis=0)
    std_distance_map = np.std(dist_maps, axis=0)


    folder = os.path.join(trajectory_path, "Analysis")
    os.makedirs(folder, exist_ok=True)  

    avg_path = os.path.join(folder, "average_distance_map.npy")
    std_path = os.path.join(folder, "std_distance_map.npy")

    np.save(avg_path, avg_distance_map)
    np.save(std_path, std_distance_map)

    return avg_distance_map, std_distance_map





def average_run_distance_maps(trajectory_folder):
    """
    Average all average_distance_map.npy files found in run folders
    inside a trajectory folder.

    Parameters
    ----------
    trajectory_folder : str
        Path to the trajectory folder containing run_* directories.

    Returns
    -------
    grand_avg_map : ndarray
    grand_std_map : ndarray
    """

    maps = []

    # Find all run folders
    run_folders = sorted(
        d for d in os.listdir(trajectory_folder)
        if os.path.isdir(os.path.join(trajectory_folder, d))
        and d.startswith("run_")
    )

    if len(run_folders) == 0:
        raise ValueError(f"No run folders found in {trajectory_folder}")

    for run_folder in run_folders:

        map_path = os.path.join(
                    trajectory_folder,
                    run_folder,
                    "Analysis",
                    "average_distance_map.npy"
                )

        if os.path.exists(map_path):
            maps.append(np.load(map_path))
        else:
            print(f"Warning: {map_path} not found")

    if len(maps) == 0:
        raise ValueError("No average_distance_map.npy files found")

    maps = np.array(maps)

    grand_avg_map = np.mean(maps, axis=0)
    grand_std_map = np.std(maps, axis=0)

    # Save directly in the trajectory folder
    folder = os.path.join(trajectory_folder, "Analysis")
    os.makedirs(folder, exist_ok=True)

    np.save(
        os.path.join(folder,
                     "average_distance_map_all_runs.npy"),
        grand_avg_map
    )

    np.save(
        os.path.join(folder,
                     "std_distance_map_all_runs.npy"),
        grand_std_map
    )

    return grand_avg_map, grand_std_map









# ------------------------------
#  Function for Distance Scaling 
# ------------------------------



def round_to_nearest_multiple(number, multiple):
    return np.round(number / multiple) * multiple

def calc_distances(traces, region, cell_type, crop_gpos = False, min_length = 30):
    '''
    It takes chromosome tracing data (real cells) and converts it into:
    (physical distance, genomic distance) to plot distance vs genomic separation 
    and compare directly to simulations
    '''
    
    print(cell_type, region)
    df = traces.loc[(traces.region==region) & (traces.cell_type==cell_type) & (traces.g_pos>0)]
    #Filters specific chromosome region (e.g. Chr2), specific cell type (e.g. IMR90), valid genomic positions

    # A small correction needed due to an error in some traces
    if region == 'Chr14_full':
        full_count = 84
    elif region == 'Chr2_full':
        full_count = 239

    df = df[df.trace_id.isin(list(df.trace_id.unique()[df.groupby('trace_id').count().fit_id==full_count]))] 
    # only keep cells where all expected loci are detected

    print("Number of traces used: " + str(len(df.trace_id.unique())))
    if crop_gpos == True: #Only uses "q-arm" of chromosome 2 for that data. Turn off for other regions.
        df = df.query('g_pos > 120000000')
        # Optional biological filtering. Examples: keep only q-arm of chromosome, remove centromere region, etc.
    else:
            pass
    
    df = tr.tracing_length_qc(df, min_length=min_length)
    # Quality control:Removes: remove short traces, noisy / incomplete data
    
    pwds = tr.pwd_calc(df,)
    g_dists = np.repeat(squareform(pdist(df.groupby('trace_id')[['g_pos']].get_group(df.trace_id.unique()[0]).values))[None,:,:], repeats=pwds.shape[0],axis=0) #Read genomic distances from probes list and match to pwd matrix.
    dists = pd.DataFrame(np.stack([pwds.ravel(), g_dists.ravel()]).T, columns=['dist', 'g_dist'])
    dists['region'] = region
    dists['cell_type'] = cell_type

    if region in ['Chr2_1MB', 'Chr5_1MB', 'Chr14_1MB']:
        dists['g_dist'] = round_to_nearest_multiple(dists['g_dist']/1000,12)
    elif region in ['Chr2_12Mb', 'Chr14_11Mb','Chr18_13Mb']:
        dists['g_dist'] = round_to_nearest_multiple(dists['g_dist']/1000,200)
    elif region in ['Chr2_full', 'Chr14_full']:
        dists['g_dist'] = round_to_nearest_multiple(dists['g_dist']/1000,1000)
    return dists




def load_and_compute_genomic_stats(
    main_folder,
    bead_size_kb,
    eq=2000,
    jump=1000,
    stride=4,
    save_name="df_stats.csv",
    save_per_run=True,
    save_global=True
):

    runs = []


    # Load runs 
    for sub in sorted(os.listdir(main_folder)):
        run_path = os.path.join(main_folder, sub)

        if not os.path.isdir(run_path):
            continue

        files = [f for f in os.listdir(run_path) if f.endswith(".npy")]
        if not files:
            continue

        arr = np.load(os.path.join(run_path, files[0]))

        if arr.ndim != 3:
            continue

        arr = arr[eq::jump]
        runs.append((sub, arr))

    per_run_results = []
    all_run_stats = []

    # Process each run
    for run_name, traj in runs:

        run_g = []
        run_d = []

        for frame in traj:

            coords = frame[::stride]
            N = coords.shape[0]

            dist_matrix = compute_distance_map(coords)

            i, j = np.triu_indices(N, k=1)

            run_d.append(dist_matrix[i, j])
            run_g.append((j - i) * bead_size_kb * stride)

        run_g = np.concatenate(run_g)
        run_d = np.concatenate(run_d)

        df_run = pd.DataFrame({
            "g_dist": run_g,
            "dist": run_d
        })

        df_run_stats = df_run.groupby("g_dist")["dist"].agg(
            dist_mean="mean",
            dist_std="std",
            count="count"
        ).reset_index()

        per_run_results.append((run_name, df_run_stats))
        all_run_stats.append(df_run_stats)


        # Save per-run stats
        if save_per_run:
            run_folder = os.path.join(main_folder, run_name)
            analysis_folder = os.path.join(run_folder, "Analysis")
            os.makedirs(analysis_folder, exist_ok=True)

            df_run_stats.to_csv(
                os.path.join(analysis_folder, save_name),
                index=False
            )

    # Global stats
    df_all = pd.concat(all_run_stats, ignore_index=True)

    df_global_stats = df_all.groupby("g_dist").apply(
        lambda x: pd.Series({
            "dist_mean": np.average(x["dist_mean"], weights=x["count"]),
            "dist_std": np.sqrt(np.average(x["dist_std"]**2, weights=x["count"])),
            "count": x["count"].sum()
        })
    ).reset_index()


    # Save global stats
    if save_global:
        analysis_folder = os.path.join(main_folder, "Analysis")
        os.makedirs(analysis_folder, exist_ok=True)

        df_global_stats.to_csv(
            os.path.join(analysis_folder, save_name),
            index=False
        )

    return per_run_results, df_global_stats







# ---------------------------------------------------------------------------------------------------------------------------------------------
# Multi-Polymer Simulations
# ----------------------------------------------------------------------------------------------------------------------------------------------




# ----------------------------------------
# Intra- and Interchromosomal contact map
# ----------------------------------------


def compute_contact_map_multipolymer(
    traj_path,
    seq_path,
    cutoff=10.0,
    k=10,
    frame_start=1000,
    frame_stride=1000
):

    # Load trajectory
    coords = np.load(traj_path)

    print("Trajectory shape:", coords.shape)

    # Load sequence
    with open(seq_path, "r") as f:
        data = json.load(f)

    n_polymers = data["n_polymers"]
    L = data["L"]

    print(f"Polymers: {n_polymers}, L: {L}")

    # Downsample
    L_new = L // k

    downsampled = []

    for p in range(n_polymers):
        start = p * L
        end = (p + 1) * L

        polymer = coords[:, start:end, :]
        polymer = polymer[:, ::k, :]

        downsampled.append(polymer)

    coords = np.concatenate(downsampled, axis=1)
    L = L_new

    print(f"Downsampled L: {L}")

    # Create full matrix
    N = n_polymers * L

    full_map = np.zeros((N, N))
    frame_indices = np.arange(frame_start, coords.shape[0], frame_stride)
    n_frames = len(frame_indices)

    # Loop over frames
    for frame in frame_indices:
        pos = coords[frame]

        # Split polymers
        blocks = [
            pos[p * L:(p + 1) * L]
            for p in range(n_polymers)]

        # Pairwise polymer contacts
        for i in range(n_polymers):
            for j in range(n_polymers):
                a = blocks[i]
                b = blocks[j]

                dist = np.linalg.norm(
                    a[:, None, :] - b[None, :, :],
                    axis=-1
                )

                contacts = (dist < cutoff).astype(float)

                full_map[
                    i * L:(i + 1) * L,
                    j * L:(j + 1) * L
                ] += contacts

    full_map /= n_frames


    # after full_map is computed
    run_folder = os.path.dirname(traj_path)
    system_folder = os.path.dirname(run_folder)

    analysis_dir = os.path.join(system_folder, "Analysis")
    os.makedirs(analysis_dir, exist_ok=True)

    out_path = os.path.join(analysis_dir, "contact_map.npz")

    np.savez(
        out_path,
        full_map=full_map,
        n_polymers=n_polymers,
        L=L,
        cutoff=cutoff,
        k=k
    )

    print(f"Saved contact map to: {out_path}")






# ----------------------------
# Computing pairwise distances
# -----------------------------



def compute_and_save_multi_polymer_distance_maps(
    traj_file,
    eq=2000,
    jump=1000,
    output_polymer_name="polymer_distance_maps.npz",
    output_global_name="global_distance_map.npz"
):

    # Load sequence metadata
    folder = os.path.dirname(traj_file)
    with open(os.path.join(folder, "sequence.json"), "r") as f:
        seq_data = json.load(f)

    L = seq_data["L"]
    n_polymers = seq_data["n_polymers"]

    print(f"Processing: {traj_file}")
    print(f"L={L}, polymers={n_polymers}")

    # Load trajectory array
    traj = np.load(traj_file)

    polymer_maps = []

    # Compute average distance map per polymer
    for p in range(n_polymers):

        print(f"Polymer {p}")

        start = p * L
        end = (p + 1) * L

        # Extract polymer trajectory
        traj_poly = traj[:, start:end, :]

        # Apply equilibration and frame skipping
        subset_traj = traj_poly[eq::jump]

        # Compute distance matrices for each frame
        sim_matrices = [
            compute_distance_map(frame) for frame in subset_traj
        ]

        # Average over time
        sim_avg = np.nanmean(sim_matrices, axis=0)

        polymer_maps.append(sim_avg)

    # Stack polymer results into one array
    polymer_maps = np.array(polymer_maps)

    # Compute global average across polymers
    global_avg = np.nanmean(polymer_maps, axis=0)

    # Base folders
    run_folder = os.path.dirname(traj_file)
    parent_folder = os.path.dirname(run_folder)


    # Save polymer-specific maps 
    run_analysis_dir = os.path.join(run_folder, "Analysis")
    os.makedirs(run_analysis_dir, exist_ok=True)

    poly_path = os.path.join(run_analysis_dir, output_polymer_name)

    np.savez(
        poly_path,
        polymer_maps=polymer_maps,
        L=L,
        n_polymers=n_polymers,
        eq=eq,
        jump=jump
    )

    print(f"Saved polymer maps to: {poly_path}")


    # Save global map 
    parent_analysis_dir = os.path.join(parent_folder, "Analysis")
    os.makedirs(parent_analysis_dir, exist_ok=True)

    global_path = os.path.join(parent_analysis_dir, output_global_name)

    np.savez(
        global_path,
        global_avg=global_avg,
        L=L,
        n_polymers=n_polymers,
        eq=eq,
        jump=jump
    )

    print(f"Saved global map to: {global_path}")






# -----------------
# Distance scaling 
# ----------------
import os
import json
import numpy as np
import pandas as pd


def load_and_compute_genomic_stats_multipolymers(
    traj_file,
    bead_size_kb,
    eq=2000,
    jump=1000,
    stride=4,
    output_per_polymer_name="sim_per_polymer_stats.csv",
    output_global_name="sim_global_stats.csv"
):

    folder = os.path.dirname(traj_file)

    with open(os.path.join(folder, "sequence.json"), "r") as f:
        seq = json.load(f)

    L = seq["L"]
    n_polymers = seq["n_polymers"]

    traj = np.load(traj_file)
    traj = traj[eq::jump]

    all_g = []
    all_d = []
    per_polymer_results = []

    for p in range(n_polymers):

        start = p * L
        end = (p + 1) * L

        traj_poly = traj[:, start:end]

        run_g = []
        run_d = []

        for frame in traj_poly:

            coords = frame[::stride]
            N = coords.shape[0]

            dist_matrix = np.linalg.norm(
                coords[:, None, :] - coords[None, :, :],
                axis=-1
            )

            i, j = np.triu_indices(N, k=1)

            run_d.append(dist_matrix[i, j])
            run_g.append((j - i) * bead_size_kb * stride)

        run_g = np.concatenate(run_g)
        run_d = np.concatenate(run_d)

        df_run = pd.DataFrame({
            "g_dist": run_g,
            "dist": run_d
        })

        df_stats = df_run.groupby("g_dist")["dist"].agg(
            dist_mean="mean",
            dist_std="std",
            count="count"
        ).reset_index()

        per_polymer_results.append(df_stats)

        all_g.append(run_g)
        all_d.append(run_d)

    # Global
    all_g = np.concatenate(all_g)
    all_d = np.concatenate(all_d)

    df_global = pd.DataFrame({
        "g_dist": all_g,
        "dist": all_d
    })

    df_global_stats = df_global.groupby("g_dist")["dist"].agg(
        dist_mean="mean",
        dist_std="std",
        count="count"
    ).reset_index()

    # Save
    run_folder = os.path.dirname(traj_file)
    parent_folder = os.path.dirname(run_folder)

    run_analysis_dir = os.path.join(run_folder, "Analysis")
    parent_analysis_dir = os.path.join(parent_folder, "Analysis")

    os.makedirs(run_analysis_dir, exist_ok=True)
    os.makedirs(parent_analysis_dir, exist_ok=True)

    out_poly = os.path.join(run_analysis_dir, output_per_polymer_name)

    df_poly_all = pd.concat(
        per_polymer_results,
        keys=range(n_polymers),
        names=["polymer"]
    ).reset_index(level=0)

    df_poly_all.to_csv(out_poly, index=False)

    out_global = os.path.join(parent_analysis_dir, output_global_name)
    df_global_stats.to_csv(out_global, index=False)

    print(f"Saved per-polymer stats: {out_poly}")
    print(f"Saved global stats: {out_global}")

    return out_poly, out_global



# ----------------
# Polymer Overlap
# ----------------


def compute_polymer_overlap(
    systems,
    label_dict,
    n_monomers,
    polymer_ids,
    equilibration,
    jump,
    output_path
):
    """
    Computes overlap ratios and saves results as .npy structured array.
    System metadata (Ae, B, rho) is NOT stored here — only used for labeling if needed.
    """

    all_data = []

    for system_name, path in systems.items():

        traj = np.load(path)
        traj_sel = traj[equilibration::jump]

        COM_mean = {}
        Rg_mean = {}

        # Compute COM and Rg per polymer
        for p in polymer_ids:

            start = p * n_monomers
            end = (p + 1) * n_monomers

            coords = traj_sel[:, start:end, :]

            com = coords.mean(axis=1)
            COM_mean[p] = com.mean(axis=0)

            diff = coords - com[:, None, :]
            rg = np.sqrt(np.mean(np.sum(diff**2, axis=2), axis=1))
            Rg_mean[p] = rg.mean()

        # Pairwise ratios
        for i, j in itertools.combinations(polymer_ids, 2):

            d_ij = np.linalg.norm(COM_mean[i] - COM_mean[j])
            ratio = d_ij / (Rg_mean[i] + Rg_mean[j])

            all_data.append((system_name, f"{i}-{j}", ratio))

    # Structured array (ONLY physics data)
    dtype = np.dtype([
        ("system", "U50"),
        ("pair", "U20"),
        ("ratio", "f8")
    ])

    all_array = np.array(all_data, dtype=dtype)

    # Save
    first_path = next(iter(systems.values()))
    run_folder = os.path.dirname(first_path)
    parent_folder = os.path.dirname(run_folder)

    analysis_dir = os.path.join(parent_folder, "Analysis")
    os.makedirs(analysis_dir, exist_ok=True)

    final_path = os.path.join(analysis_dir, os.path.basename(output_path))

    np.save(final_path, all_array)

    print(f"Saved results to {final_path}")

    return all_array