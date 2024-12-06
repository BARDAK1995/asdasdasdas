import scipy
import scipy.integrate
from matplotlib import animation
from IPython.display import HTML
from matplotlib import pyplot as plt
from pydmd import DMD
import os
import pandas as pd
from pydmd.plotter import plot_summary
import numpy as np
from dmd_functions import *
from lst_functions import *

import glob

xdom=60
ydom=2.5
full_column_length = 75  # You may need to adjust this value
OFFSET = 40

caseName = "expensive_Ref/"

folderLocation="Datas/" + caseName 
VisualMode = False
# Get all .npy files in the folder
npy_files = glob.glob(os.path.join(folderLocation, '*.npy'))
# Load each file
data_dict = {}
filename_list=[]

probes_filename = 'probes.data'
for file_path in npy_files:
    # Get the file name without extension
    file_name = os.path.splitext(os.path.basename(file_path))[0]
    folder_path = os.path.join(folderLocation, file_name)
    video_output_dir = os.path.join(folderLocation, file_name, "videos")
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    if not os.path.exists(video_output_dir):
        os.makedirs(video_output_dir)
    
    snapshotsNp= np.load(file_path)[3000:,:,:].transpose(0, 2, 1)
    # snapshotsNp= np.load(file_path)[3000:,:,:]

    mean_field = snapshotsNp.mean(axis=0)  # Shape: (y_dim, x_dim)
    moving_avg= calculate_moving_average_cumsum(snapshotsNp)
    snapshotsNp = snapshotsNp - moving_avg     #mean-substracted unsteady flow variables

    x_probes, y_probes = read_data_probeLocation(probes_filename)

    snapshots = [snapshotsNp[i] for i in range(snapshotsNp.shape[0])]
    X_DIM = snapshots[0].shape[1]; Y_DIM = snapshots[0].shape[0]
    x1 = np.linspace(0, xdom, X_DIM)  # 200 points for length 12
    y2 = np.linspace(0, ydom, Y_DIM)  # 50 points for height 3
    x1grid, y2grid = np.meshgrid(x1, y2)
    time = np.linspace(0, snapshotsNp.shape[0], snapshotsNp.shape[0]+1)#micrsosec
    dt = time[1]-time[0]

    plot_svd_distribution(snapshots, folder_path)
    # # PLOT SOME FOR DEBUG______________________________________________________________________________
    plot_flow_field(
        snapshotsNp=snapshotsNp,
        mean_field=mean_field,
        Probe_locations = (x_probes, y_probes),
        x1=x1,
        y1=y2,
        plotlocation=folder_path,
        chosen_cmap='viridis',  
        offset = OFFSET
    )

    # DMD ANALYSIS______________________________________________________________________________
    svd_rank = 30
    dmd = DMD(svd_rank=svd_rank, tlsq_rank=0, exact=True, opt=True,forward_backward=True,sorted_eigs='abs')
    dmd.fit(snapshots)

    dmd.dmd_time['tend'] = time[-1]
    dmd.dmd_time['dt'] = dt
    dmd_states = [state.reshape(x1grid.shape) for state in dmd.reconstructed_data.T]
    # aspect_ratio = x1grid.shape[1] / x1grid.shape[0]
    # fig_width = 10
    # fig_height = fig_width / aspect_ratio
    x_lim = (100, 550)    # Example x-axis limits in kHz
    y_lim = (5, 4000)     # Example y-axis limits for amplitude
    plot_dmd_frequencies(
        dmd=dmd,
        x_lim=x_lim,
        y_lim=y_lim,
        plotlocation=folder_path,
        width=50,                # Optional: Adjust bar width
        figsize=(12, 4),         # Optional: Adjust figure size
        color='grey',            # Optional: Set bar color
        edgecolor='black',       # Optional: Set bar edge color
        save_dpi=300             # Optional: Set resolution of saved plot
    )
    

    plot_summary(
        dmd,
        x=x1,
        y=y2,
        t=dmd.dmd_timesteps,
        d=1,
        continuous=False,
        snapshots_shape=(X_DIM,Y_DIM),  # This is correct
        index_modes=(0, 1, 2),
        filename=folderLocation + file_name +"/" + "summary123.png",
        figsize=(20, 8),  # Adjusted for better visibility of the wide aspect ratio
        dpi=200,
        mode_cmap='RdBu_r',
        max_eig_ms=15,
        title_fontsize=16,
        label_fontsize=14,
        plot_semilogy=True,
        order='F'  # Try 'F' order if 'C' doesn't work
    )
    
    total_frames = len(dmd_states)-1
    duration = 20 #seconds
    fps = total_frames / duration
    time_per_frame = 1e-7

    plot_summaryNEW2(
        dmd,
        x=x1,  # x1 should be your array of x-values in meters
        y=y2,  # y2 should be your array of y-values in meters
        t=dmd.dmd_timesteps,
        snapshots_shape=(Y_DIM, X_DIM),
        modes_per_plot=3,
        filename=folderLocation + file_name + "/Detail.png",
        figsize=(20, 8),
        dpi=200,
        mode_cmap='RdBu_r',
        title_fontsize=16,
        label_fontsize=14,
        order='C',
        time_per_frame=time_per_frame,  # Pass the time per frame
        Probe_locations=(x_probes, y_probes),  # Specify the probes data file
        offset = OFFSET
    )
    plot_summaryNEW_NonDim(
        dmd,
        x=x1,  # x1 should be your array of x-values in meters
        y=y2,  # y2 should be your array of y-values in meters
        t=dmd.dmd_timesteps,
        snapshots_shape=(Y_DIM, X_DIM),
        modes_per_plot=3,
        filename=folderLocation + file_name + "/NondimDetail.png",
        figsize=(20, 8),
        dpi=200,
        mode_cmap='RdBu_r',
        title_fontsize=16,
        label_fontsize=14,
        order='C',
        time_per_frame=time_per_frame,  # Pass the time per frame
        Probe_locations=(x_probes, y_probes),  # Specify the probes data file
        offset = OFFSET
    )
    
    targetModes = 4
    if file_name == "U_snapshots": 
        kernel_size_y = 11
        scalar = 0.3
    elif file_name == "P_snapshots": 
        kernel_size_y = 11
        scalar = 0.06
        targetModes=5
    else:
        kernel_size_y = 9
        scalar = 1
    targetModes = 8

    for mode in range(targetModes):
        plot_dmd_mode_amplitude_with_probe(
        dmd=dmd,
        mode_idx=mode,  # Index among positive frequency modes
        x=x1,
        y=y2,
        snapshots_shape=(Y_DIM, X_DIM),
        filename=f'mode_{mode+1}_amplitude.png',
        x_smooth=x_probes,
        y_smooth=y_probes*scalar,
        offset=OFFSET,
        kernel_size_x=13,  # Averaging span in x-direction
        kernel_size_y=kernel_size_y,   # Averaging span in y-direction
        plotlocation= folderLocation + file_name + "/Amp_new/",
        visualmode=True
        )
        plot_dmd_mode_amplitude_with_probe_NONDIM(
        dmd=dmd,
        mode_idx=mode,
        x=x1,                  # x in mm
        y=y2,                  # y in mm
        snapshots_shape=(Y_DIM, X_DIM),
        filename=f'mode_{mode+1}_amplitude_NONDIM.png',
        x_smooth=x_probes,     # x_probes in mm
        y_smooth=y_probes*scalar,  # y_probes in mm
        offset=OFFSET,         # offset in mm
        kernel_size_x=13/9,
        kernel_size_y=kernel_size_y/9,
        plotlocation=folderLocation + file_name + "/NonDim_Amp_new/",
        visualmode=True,
        BL_blasiusFile="probes.data"
    )
    if VisualMode:
        # create_animation(  
        #     data=dmd_states, 
        #     x1grid=x1grid, 
        #     y2grid=y2grid, 
        #     video_output_dir=video_output_dir,
        #     plot_name='DMD Reconstruction',
        #     update_func=update,
        #     display_video=True,
        #     save_video=True
        # )
        create_comparison_animation(
            snapshots=snapshots,
            dmd_states=dmd_states,
            x1grid=x1grid,
            y2grid=y2grid,
            video_output_dir=video_output_dir,
            plot_name='Ndensity',
            nModes=svd_rank,  # Assuming svd_rank is defined elsewhere in your code
            display_video=True,
            save_video=True
        )


# file_path_2ndMode = 'blasius\\lstup2.csv'
file_path_2ndMode = os.path.join('blasius', 'lstup2.csv')

dict2Mode = get_2ndMode_reference(file_path_2ndMode, Re_nondim = (11.76*10**6),U_edge = 857,phase_speed = 0.91)
X_example_array = np.array([60, 70, 90])/1000  # Array-like input
unstableFrequencies_atLocalX_plot(dict2Mode, X_example_array)
unstableF_atLocalR_nondim_plot(dict2Mode, X_example_array)


unstableXlocations_atFreq_plot(dict2Mode, target_frequencies=210.0)


mode_order = np.argsort(-np.abs(dmd.amplitudes))
frequencies = dmd.frequency[mode_order]  # Frequencies in Hz
# Convert frequencies to kHz for display
frequencies_kHz = frequencies * 10000 # Convert Hz to kHz
positive_freq_indices = np.where(frequencies > 0)[0]
frequencies = frequencies[positive_freq_indices]
frequencies_kHz = frequencies_kHz[positive_freq_indices]
unstableXlocations_atFreq_plot(dict2Mode, target_frequencies=frequencies_kHz[:4])
unstableRlocations_at_nondimF_plot(dict2Mode, target_frequencies=frequencies_kHz[:4])

W_lower_vs_X = dict2Mode['w_kHz_lower_vs_X']  # Tuple (R_lower, F_lower)
W_upper_vs_X = dict2Mode['w_kHz_upper_vs_X']  # Tuple (R_upper, F_upper)
X_min, X_max = get_R_bounds(W_lower_vs_X, W_upper_vs_X, frequencies_kHz[:4]*1000)
X_min*=1000;X_max*=1000
