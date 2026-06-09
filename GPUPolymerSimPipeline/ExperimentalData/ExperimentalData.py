"""
-----------------------------------------
Functions for Handling Experimental Data 
-----------------------------------------

This module contains functions for importing, preprocessing, analyzing,
and visualizing experimental Hi-C datasets and chromatin tracing.

"""

# Import modules
import cooler
import os
import sys
import json
import numpy as np
import pandas as pd
import seaborn as sns  


import matplotlib 
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


# For importing LoopTrace when using IDUN:
sys.path.append("/cluster/work/andrhsta/Masteroppgave/TracingData")
sys.path.append("/cluster/work/andrhsta/Masteroppgave/TracingData/TraceAnalysis")

from looptrace import trace_analysis_functions as tr # Custom functions for analyzing tracing data
from scipy.spatial.distance import pdist, squareform




# Nice font in plots
def apply_serif_font():
    import matplotlib.pyplot as plt
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["DejaVu Serif"]


apply_serif_font()





# -------------------------------------
# Plotting Compartment A/B bin sequence 
# ------------------------------------- 

def plot_compartment_track_40kb(csv_file,
                                compartment_colors={'A':"#053061", 'B': "darkred"},
                                output_folder="CompartmentPlots",
                                filename=None,
                                json_filename=None,  # <- optional JSON output
                                figsize=(20, 2),
                                segment_start=None,
                                segment_end=None):
    """
    Plot genomic compartments (A/B) specifically for the IMR90 40kb resolution CSV.
    Works directly with the tab-separated CSV with columns:
    'chr', 'pos_start', 'pos_end', 'compartment', 'compartment_score'.
    """

    # Load CSV correctly
    df = pd.read_csv(csv_file, sep='\t', header=0)
    if len(df.columns) == 1 and '\t' in df.columns[0]:
        df = df[df.columns[0]].str.split('\t', expand=True)
        df.columns = ['chr', 'pos_start', 'pos_end', 'compartment', 'compartment_score']

    df.columns = ['chr', 'pos_start', 'pos_end', 'compartment', 'compartment_score']

    # Strip whitespace and convert positions
    df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
    df['pos_start'] = pd.to_numeric(df['pos_start'], errors='coerce')
    df['pos_end'] = pd.to_numeric(df['pos_end'], errors='coerce')
    df = df.dropna(subset=['pos_start', 'pos_end'])
    df = df[['chr', 'pos_start', 'pos_end', 'compartment']].copy()
    df = df.sort_values('pos_start').reset_index(drop=True)


    # Generate full 40kb bin coverage
    bin_size = 40000
    chrom_start = df['pos_start'].min()
    chrom_end = df['pos_end'].max()

    full_bins = pd.DataFrame({'pos_start': range(int(chrom_start), int(chrom_end), bin_size)})
    full_bins['pos_end'] = full_bins['pos_start'] + bin_size
    full_bins['chr'] = df['chr'].iloc[0]
    full_bins['compartment'] = None

    for _, row in df.iterrows():
        mask = (full_bins['pos_start'] >= row['pos_start']) & (full_bins['pos_end'] <= row['pos_end'])
        full_bins.loc[mask, 'compartment'] = row['compartment']

    # Fill missing compartments with 'A'
    full_bins['compartment'] = full_bins['compartment'].fillna('A')
    df = full_bins.copy()
    print(f"Total number of bins: {len(df)}")

    # Count A and B bins
    n_A = (df['compartment'] == 'A').sum()
    n_B = (df['compartment'] == 'B').sum()

    print(f"Number of A bins: {n_A}")
    print(f"Number of B bins: {n_B}")
    print(f"Total bins (check): {n_A + n_B}")


    # Save JSON sequence
    if json_filename is None:
        json_filename = (filename.replace('.png', '_sequence.json') 
                         if filename else "CompartmentTrack_40kb_sequence.json")
    os.makedirs(output_folder, exist_ok=True)
    sequence_path = os.path.join(output_folder, json_filename)
    with open(sequence_path, 'w') as f:
        json.dump(df['compartment'].tolist(), f)
    print(f"Compartment sequence saved to {sequence_path}")


    # Filter segment if requested
    if segment_start is not None:
        df = df[df['pos_start'] >= segment_start]
    if segment_end is not None:
        df = df[df['pos_end'] <= segment_end]


    # Plot compartments
    fig, ax = plt.subplots(figsize=figsize)
    plt.subplots_adjust(top=0.9)

    for _, row in df.iterrows():
        ax.bar(x=row['pos_start'],
               height=1,
               width=row['pos_end'] - row['pos_start'],
               color=compartment_colors.get(row['compartment'], 'white'),
               align='edge')

    ax.set_xlim(df['pos_start'].min(), df['pos_end'].max())
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_xlabel('Genomic position (bp)', fontsize=26)
    ax.tick_params(axis='x', labelsize=32)


    # Save plot
    if filename is None:
        filename = "Colors_CompartmentTrack_40kb.png"
    fig.savefig(os.path.join(output_folder, filename), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Plot saved to {os.path.join(output_folder, filename)}")









# --------------------------------------
# Plotting Hi-C data: Linear and Log Map
# --------------------------------------


def plot_hic_heatmap(
    cool_file,
    chromosome="2",
    output_folder="hic_plots",
    filename="hic_heatmap.png",
    balance=False,
    figsize=(68, 24),
    cmap_name="custom"
):
    """
    Plot Hi-C contact map in linear and log scale from a .cool file.
    
    Parameters:
        cool_file (str): Path to .cool file
        chromosome (str): Chromosome to visualize (e.g. "1", "2", "chr1")
        output_folder (str): Folder to save figure
        filename (str): Output image filename
        balance (bool): Whether to use balanced matrix
        figsize (tuple): Figure size
        cmap_name (str): Colormap name (currently uses fixed custom palette)
    """

    # Load data
    c = cooler.Cooler(cool_file)
    matrix = c.matrix(balance=balance).fetch(chromosome)

    print("Matrix shape:", matrix.shape)

  
    # Colormap
    colors = [
        "#053061",
        "#2166AC",
        "#92C5DE",
        "#F7F7F7",
        "#DA8683",
        "#DC6472",
        "#F70404"
    ]

    custom_cmap = mcolors.LinearSegmentedColormap.from_list(
        cmap_name,
        colors
    )

  
    # Figure setup
    fig = plt.figure(figsize=figsize)

  
    # Linear scale
    ax1 = fig.add_subplot(121)

    im1 = ax1.imshow(matrix, cmap=custom_cmap, origin="upper")

    ax1.set_xlabel(f"Chromosome {chromosome} Position (bins)", fontsize=30)
    ax1.set_ylabel(f"Chromosome {chromosome} Position (bins)", fontsize=30)
    ax1.tick_params(axis="both", labelsize=20)

    cbar1 = plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    cbar1.set_label("Contact Count", fontsize=20)
    cbar1.ax.tick_params(labelsize=15)

  
    # Log scale
    ax2 = fig.add_subplot(122)

    log_matrix = np.log1p(matrix)

    im2 = ax2.imshow(
        log_matrix,
        cmap=custom_cmap,
        origin="upper",
    )

    im2.set_clim(0, 10)

    ax2.set_xlabel(f"Chromosome {chromosome} Position (bins)", fontsize=30)
    ax2.set_ylabel(f"Chromosome {chromosome} Position (bins)", fontsize=30)
    ax2.tick_params(axis="both", labelsize=20)

    cbar2 = plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    cbar2.set_label("log1p(Contact Count)", fontsize=20)
    cbar2.ax.tick_params(labelsize=15)

  
    # Layout + save
    plt.tight_layout()

    os.makedirs(output_folder, exist_ok=True)
    save_path = os.path.join(output_folder, filename)

    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved figure to: {save_path}")










# ----------------------------------
# Chromatin Tracing Distance Matrix
# ----------------------------------

def replace_nan_with_diagonal_median(matrix):
    # Interpolates missing values which are not allowed in some analysis
    # Get the shape of the matrix

    """
    Some downstream analyses (e.g. PCA) cannot handle NaNs.

    This fills NaNs with the median of their diagonal because diagonals represent genomic separation.
    So missing values are replaced with the typical value for that separation.
    """
    rows, cols = matrix.shape
    # Iterate over each diagonal
    for diag in range(-rows + 1, cols):
        # Extract the diagonal elements
        diag_elements = np.diagonal(matrix, offset=diag)
        # Calculate the median of non-NaN elements in the diagonal
        median_value = np.nanmedian(diag_elements)
        # Replace NaN values in the diagonal with the median value
        diag_indices = np.where(np.isnan(diag_elements))
        for index in diag_indices[0]:
            if diag >= 0:
                matrix[index, index + diag] = median_value
            else:
                matrix[index - diag, index] = median_value
    
    return matrix           # Replace each NaN in that diagonal with the median. Same matrix, but with NaNs filled.
 



def plot_trace_distance_matrix():
    traces = pd.read_csv(
    "/cluster/work/andrhsta/Masteroppgave/TracingData/TraceAnalysis/combined_dataset_v2.csv.gz"
    )

    traces['tmp'] = traces['exp'] + '_' + traces['region'] + '_' + traces['trace_id'].astype(str)
    trace_id_map = {tmp:i for i,tmp in enumerate(traces.tmp.unique())}
    traces['trace_id'] = traces['tmp'].map(trace_id_map)

    region = 'Chr2_full' #Select a region. For full chromosomes, we have "Chr2_full" and "Chr14_full"
    cell_type = 'IMR90_ctrl' #Select correct cell type

    df = traces.loc[(traces.region==region) & (traces.cell_type==cell_type) & (traces.g_pos>0)] # Filter correct region and only valid genomic positions
    # Match chosen chromosome, chosen cell type, valid genomic position (g_pos>0 means valid position, not missing data)

    # A small correction needed due to an error in some traces
    if region == 'Chr14_full': # Expected number of loci per full chromosome.
        full_count = 84
    elif region == 'Chr2_full':
        full_count = 239


    # Group by trace_id, count rows per trace, keeps only traces that have exactly full_count loci
    df = df[df.trace_id.isin(list(df.trace_id.unique()[df.groupby('trace_id').count().fit_id==full_count]))] 

    view_slice = slice(None, None)

    # Removes traces that: have too short spatial length, are badly reconstructed, fail geometric QC
    df = tr.tracing_length_qc(df, min_length=30) # Filter out low-quality traces


    # Computes per-trace distance matrices
    pwds = tr.pwd_calc(df) #Calculate pairwise-distance matrix of traces
    print(pwds.shape) # Each trace → one square distance matrix, (n_traces, n_bins, n_bins)

    # Population median distance matrix
    obs_dists = np.nanmedian(pwds, axis=0)  # Calculate median pairwise distances of all traces.
    obs_dists = replace_nan_with_diagonal_median(obs_dists) # Replace nan values as later calculations cannot handle.
    #obs_dists = median_filter(obs_dists) #Cleanup data to avoid emphasizing noise


    # For single frames of experimental data
    # obs_dists = pwds[554]   # first frame / first trace distance matrix
    # obs_dists = replace_nan_with_diagonal_median(obs_dists)

    # Optional: check for NaNs
    if np.isnan(obs_dists).any():
        print("Warning: NaNs present in the median distance map!")

    # Plot the raw pairwise distance matrix
    fig, ax = plt.subplots(figsize=(6,6))
    im0 = ax.imshow(obs_dists, cmap='RdBu', vmin=200, vmax=3500)
    ax.set_xlabel("Bin Index", fontsize=18)
    ax.set_ylabel("Bin Index", fontsize=18)
    ax.tick_params(axis='both', which='major', labelsize=18, width=2, length=6)
    cbar = fig.colorbar(im0, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=18)

    # Save figure
    output_path = "/cluster/work/andrhsta/Masteroppgave/TracingData/TraceAnalysis/ExperimentalData/SingleTraceFrame554.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")

    plt.close(fig)










# ----------------------------------------
# Chromatin Tracing Distance Scaling Curve
# ----------------------------------------

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



def plot_experimental_distance_curve():
    #Load traces
    traces = pd.read_csv(r"/cluster/work/andrhsta/Masteroppgave/TracingData/TraceAnalysis/combined_dataset_v2.csv.gz")

    #Remap trace-ids so that there are no double-assignments
    traces['tmp'] = traces['exp'] + '_' + traces['region'] + '_' + traces['trace_id'].astype(str)
    trace_id_map = {tmp:i for i,tmp in enumerate(traces.tmp.unique())}
    traces['trace_id'] = traces['tmp'].map(trace_id_map)

    exp_dists = calc_distances(traces=traces, region='Chr2_full', cell_type='IMR90_ctrl', crop_gpos = True, min_length = 30)

    plt.figure(figsize=(6, 4))
    sns.lineplot(data = exp_dists, x='g_dist', y='dist', estimator=np.median, errorbar=None, linewidth = 3,  color="#5C2074")

    # Print value at 1 Mb (= 1000 kb)
    dist_1mb = exp_dists.loc[exp_dists['g_dist'] == 1000, 'dist'].median()
    print(f"Median spatial distance at 1 Mb (1000 kb): {dist_1mb}")


    plt.xlabel("Genomic distance (kb)", fontsize=14)
    plt.ylabel("Spatial distance", fontsize=14)

    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    plt.tight_layout()

    outpath = "exp_distance_plot.png"
    plt.savefig(outpath, dpi=300)
    plt.close()

    print("Saved figure to:", outpath)