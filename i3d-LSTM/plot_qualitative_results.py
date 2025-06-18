import numpy as np
import os

import matplotlib.pyplot as plt


# Prefering high recall
SKIP_FRAMES = 5
FLOW_MAG_THRESHOLD = 1.2
BATCH_SIZE = 1


# annotions are txt and have this format
# "S1.mp4" 2.3 3.06
# "S4.mp4" 7.3 7.05
# "S4.mp4" 5.3 9.57
def create_ear_annotation_dict(annotation_file):
    annotations = {}
    with open(annotation_file, "r") as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            video_name, start, end = line.split(" ")
            if video_name not in annotations:
                annotations[video_name] = []
            annotations[video_name].append((float(start), float(end)))
    f.close()

    return annotations

def plot_movement(movements, filename, ear_annotations, results_path_base, window_size, step_size, fps):
    # create plot
    fig, ax1 = plt.subplots(figsize=(8, 5))


    # get number of samples (windows)
    n_samples = len(movements)


    # get time per sample (window) knowig step size and window size
    time_per_sample = step_size / fps
    
    total_time = time_per_sample * n_samples
    

    # The first sample is the first window and it corresponds to t=0.5*WINDOW_SIZE/FPS
    # The next sample is the second window and it corresponds to t=1.5*WINDOW_SIZE/FPS
    # ETC
    # plot starting at t=0
    time_values = [i*time_per_sample for i in range(n_samples)]

    plt.xlim(time_values[0], time_values[-1])

    # plot the ear annotation on the same axis
    # get list of gt movement intervals from ear dict
    ear_movs_intervals = ear_annotations[filename]

    # change ear_movs_intervals so that none of the intervals overlaps. Merge overlapping intervals
    ear_movs_intervals.sort(key=lambda x: x[0])
    merged_intervals = []
    current_start, current_end = ear_movs_intervals[0]
    for start, end in ear_movs_intervals[1:]:
        if start <= current_end:
            current_end = max(current_end, end)
        else:
            merged_intervals.append((current_start, current_end))
            current_start, current_end = start, end
    merged_intervals.append((current_start, current_end))
    ear_movs_intervals = merged_intervals

    for i, (start, end) in enumerate(ear_movs_intervals):
        ax1.axvspan(start, end, color='red', alpha=0.3, label =  "_"*i + "Ear movement groundtruth")
        ax1.axvline(start, color='red', linestyle='dashed')
        ax1.axvline(end, color='red', linestyle='dashed')

  
    # movement in a binary vector of size time_values
    movement_intervals = []
    for i in range(n_samples):
        if movements[i] == 1:
            start = i
            end = i
            while end < n_samples-1 and movements[end] == 1:
                end += 1
            movement_intervals.append((start, end))

    # merge intervals that are contiguous
    merged_intervals = []
    for start, end in movement_intervals:
        if not merged_intervals or start > merged_intervals[-1][1] + 1:
            merged_intervals.append((start, end))
        else:
            merged_intervals[-1] = (merged_intervals[-1][0], end)

    
    # plot vertical spans inside movement intervals with dpotted lines on the limits
    for i, (start, end) in enumerate(merged_intervals):
        plt.axvspan(time_values[start], time_values[end], color='blue', alpha=0.3, label =  "_"*i + "Ear movement prediction")
        plt.axvline(time_values[start], color='blue', linestyle='dashed')
        plt.axvline(time_values[end], color='blue', linestyle='dashed')


   

    # plot legend
    plt.legend(loc='upper right', framealpha=1)

    # remove extension from filename
    filename_no_extension = filename.split(".")[0]
    plt.title(filename_no_extension)
    plt.xlabel("Time (s)")
    
    # create folder for plots
    os.makedirs(f"{results_path_base}/plots/{window_size}/{step_size}", exist_ok=True)

    # create folder for plots
    plt.savefig(f"{results_path_base}/plots/{window_size}/{step_size}/{filename}.pdf")
    plt.close()
