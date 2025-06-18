import numpy as np
from locutils.plot import plot_movement
import os

import matplotlib.pyplot as plt
import configs


# Prefering high recall
SKIP_FRAMES = 5
FLOW_MAG_THRESHOLD = 1.2
BATCH_SIZE = 1

# Clipping videos to smaller clips
WINDOW_SIZE = 20 # 0.8 seconds in every window
STEP_SIZE = 5 # A window every 0.2 seconds
FPS = 25

# annotions are txt and have this format
# "S1.mp4" 2.3 3.06
# "S4.mp4" 7.3 7.05
# "S4.mp4" 5.3 9.57
def create_ear_annotation_dict(annotation_file):
    """
    Create a dictionary of ear annotations from the annotation file.
    Args:
        annotation_file (str): Path to the annotation file.
    Returns:
        dict: Dictionary of ear annotations with video name as key and a list of tuples (start, end) as value.
    """
    # create empty dictionary
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

def plot_movement(movements, flow_mags, filename, ear_annotations):
    """
    Plot the movements, color histogram differences and flow magnitudes in a single plot, writing it to PDF.
    Args:
        movements (list): List of movements value as given by movement detection script.
        flow_mags (list): List of flow magnitudes.
        filename (str): Name of the video file.
        ear_annotations (dict): Dictionary of ear annotations.
    """
    # create plot
    fig, ax1 = plt.subplots(figsize=(8, 5))
    #ax2 = ax1.twinx()

    # get number of samples (windows)
    n_samples = len(movements)

    # get time per sample (window) knowig step size and window size
    time_per_sample = STEP_SIZE / FPS
    
    total_time = time_per_sample * n_samples
    print(f"Total time: {total_time}")
    
    # get number of frames
    number_of_video_frames = STEP_SIZE * n_samples + WINDOW_SIZE

    # The first sample is the first window and it corresponds to t=0.5*WINDOW_SIZE/FPS
    # The next sample is the second window and it corresponds to t=1.5*WINDOW_SIZE/FPS
    # ETC
    # plot starting at t=0
    time_values = [i*time_per_sample for i in range(n_samples)]

    # print len of each
    print(f"len time_values: {len(time_values)}")
    print(f"len movements: {len(movements)}")
    print(f"len flow_mags: {len(flow_mags)}")

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


    # get list of gt movement intervals from ear dict
    ear_movs_intervals = ear_annotations[filename]

    # create gt array of size len(time_values) with zeros
    gt_ear_movs = np.zeros(len(time_values))

    # Now we need to convert the ear movement intervals to the same scale as the movement intervals
    # We need to convert the time intervals to the same scale as the movement intervals
    # We need to convert the time intervals to the same scale as the movement intervals
    for start, end in ear_movs_intervals:
        # get start and end indexes
        start_idx = int(start * FPS / STEP_SIZE)
        end_idx = int(end * FPS / STEP_SIZE)
        # set to 1 the interval
        gt_ear_movs[start_idx:end_idx] = 1

    # gt_ear_movs in a binary vector of size time_values
    gt_intervals = []
    for i in range(len(gt_ear_movs)):
        if gt_ear_movs[i] == 1:
            start = i
            end = i
            while end < len(gt_ear_movs)-1 and gt_ear_movs[end] == 1:
                end += 1
            gt_intervals.append((start, end))
    
    # merge intervals that are contiguous
    merged_gt_intervals = []
    for start, end in gt_intervals:
        if not merged_gt_intervals or start > merged_gt_intervals[-1][1] + 1:
            merged_gt_intervals.append((start, end))
        else:
            merged_gt_intervals[-1] = (merged_gt_intervals[-1][0], end)

    # plot vertical spans inside gt movement intervals with dotted lines on the limits
    for i, (start, end) in enumerate(merged_gt_intervals):
        plt.axvspan(time_values[start], time_values[end], color='red', alpha=0.3, label =  "_"*i + "Ear movement groundtruth")
        plt.axvline(time_values[start], color='red', linestyle='dashed')
        plt.axvline(time_values[end], color='red', linestyle='dashed')
   

    # limit y axis to 0-10
    ax1.set_ylim(0, 3)
    #ax2.set_ylim(0, 10)

    ax1.set_ylabel("Mean flow vectors magnitude", color="black")
    plt.plot(time_values, flow_mags, label="Mean flow vectors magnitude", color="black")
    # add horizontal line at threshold
    plt.axhline(y=FLOW_MAG_THRESHOLD, color="black", linestyle='--', label="Flow vectors magnitude threshold")

    # plot legend
    plt.legend(loc='upper right', framealpha=1)

    # remove extension from filename
    filename_no_extension = filename.split(".")[0]
    plt.title(filename_no_extension)
    plt.xlabel("Time (s)")
    
    # create folder for plots
    plt.savefig(f"{results_path_base}/plots/{filename}.pdf")
    plt.close()

    return False

if __name__ == "__main__":

    results_path_base = f"{configs.OUTPUT_DIR}/movement_detection/qualitative_movement/25FPS"  # this is currently hardcoded todo
    annotation_file = f"{configs.DATASET_DIR}/full_videos/ear_annot.txt"

    ear_annotation_dict = create_ear_annotation_dict(annotation_file)

    os.makedirs(f"{results_path_base}/plots", exist_ok=True)

    visited_videos = []


    # get list of filenames in results_path_base
    filenames = os.listdir(results_path_base)
    # if it ends with txt
    filenames = [filename for filename in filenames if filename.endswith(".txt")]
    for filename in filenames:
        movements = []
        flow_mags = []
        # get video name from filename S1.mp4 from S1.mp4_color_diff.txt
        video_name = filename.split("_")[0]
        # if video name was already visited
        if video_name in visited_videos:
            continue
        visited_videos.append(video_name)

        for inner_filename in filenames:
            if video_name in inner_filename:
                # open txt for reading
                value_list = []
                with open(f"{results_path_base}/{inner_filename}", "r") as f:
                    # read all lines
                    lines = f.readlines()
                    for line in lines:
                        # read line as scientific notation number
                        value = float(line)
                        value_list.append(value)
                # close file
                f.close()
                
                if "flow_mag" in inner_filename:    
                    flow_mags=value_list
                else:
                    movements=value_list    
     
        if movements and  flow_mags:
            plot_movement(movements, flow_mags, video_name, ear_annotation_dict)