import matplotlib.pyplot as plt
from .time_conv import time_to_seconds
import json

def classify_bool_movement(median_magnitude, movement):
    movement_bools = [False] * len(movement)
    # Find local maxima in movement_smooth
    for i in range(1, len(movement)):
        if movement[i] > 0.3 + 2 * median_magnitude:
            movement_bools[i] = True
            # Set the N=3 surrounding values to True as well
            for n in range(1, 3):
                if i+n < len(movement_bools):
                    movement_bools[i+n] = True
                if i-n >= 0:
                    movement_bools[i-n] = True
        
    return movement_bools


def plot_movement(time_values, movement, movement_bools, annot_txt, filename, fps, skip_frames):
    # Plot the data
    plt.figure(figsize=(12, 6))

    #plt.yscale("log")   # Use logarithmic scale for better visualization
    plt.plot(time_values, movement, marker='o', linestyle='-', label="Mixed movement")

    # Read txt line by line
    with open(annot_txt, 'r') as file:
        for line in file:
            # values are whitespace separated
            video_name, start_time, end_time = line.split()
            # Remove extension from video name
            video_name = video_name.split(".")[0]
            if video_name in filename:
                # draw only vertical lines on plt, no text and paint the middle between start and end time
                plt.axvline(x=float(start_time), color='r', linestyle='--')
                plt.axvline(x=float(end_time), color='r', linestyle='--')
                plt.axvspan(float(start_time), float(end_time), color='r', alpha=0.2)
                
    
    # Plot movement boolean values. Paint the areas where movement is detected.
    first_true = 0

    for (mov_bool, i) in zip(movement_bools, range(len(movement_bools))):
        # Skip the first value
        if i == 0:
            continue

        # Get the previous and current boolean values
        previous_bool = movement_bools[i-1]
        current_bool = mov_bool

        # If the current value is True and the previous value is False, then we have a new movement detected
        if current_bool == True and previous_bool == False:
            first_true = time_values[i]
        # If the current value is False and the previous value is True, then we have the end of the movement detected
        elif current_bool == False and previous_bool == True:
            plt.axvspan(first_true, time_values[i], color='b', alpha=0.2)
            plt.axvline(x=float(first_true), color='b', linestyle='--')
            plt.axvline(x=float(time_values[i]), color='b', linestyle='--')
        # If we are at the end of the array and the current value is True, then we have a movement detected    
        elif i == len(movement_bools) - 1 and current_bool == True:
            plt.axvspan(first_true, time_values[i], color='b', alpha=0.2)
            plt.axvline(x=float(first_true), color='b', linestyle='--')
            plt.axvline(x=float(time_values[i]), color='b', linestyle='--')

    plt.title("Movement detection")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Movement (pixels)")
    plt.legend()
    plt.grid(True)

    # Save the plot as an image file
    plt.savefig(f"plots/{filename}_{fps}_{skip_frames}_plot.pdf", dpi=300)  # Save as a .pdf file with high resolution
    plt.close()