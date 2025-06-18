def convert_pred_to_time_intervals(pred_classes, chunk_size=64, fps=25):
    """
    Convert predicted classes to the time interval of the single largest
    consecutive run of 1s (scores > 0.9).

    Args:
        pred_classes (torch.Tensor): Predicted classes tensor of shape (B, T).
                                     (values assumed to be in [0,1] range)
        chunk_size (int): Number of frames per chunk.
        fps (int): Frames per second.

    Returns:
        activity_intervals (list): List of tuples (start_time, end_time).
                                   Each tuple indicates the largest interval
                                   of consecutive '1's for each sequence.
    """
    activity_intervals = []
    B, T = pred_classes.shape

    for i in range(B):
        # We'll keep track of the largest run of 1s
        largest_run_length = 0
        largest_run_start = -1

        current_run_length = 0
        current_run_start = -1

        for j in range(T):
            # Check if we are in an active region (score > 0.5)
            if pred_classes[i, j] > 0.5:
                # Start of a new run
                if current_run_start == -1:
                    current_run_start = j
                    current_run_length = 1
                else:
                    # Extend the current run
                    current_run_length += 1
            else:
                # We just ended a run; compare it to the largest run so far
                if current_run_length > largest_run_length:
                    largest_run_length = current_run_length
                    largest_run_start = current_run_start
                # Reset
                current_run_start = -1
                current_run_length = 0

        # Edge case: if the last position was part of a run, close it out
        if current_run_length > largest_run_length:
            largest_run_length = current_run_length
            largest_run_start = current_run_start

        # Convert the largest run to a time interval
        if largest_run_length == 0:
            # No 1s found
            start_time = 0.0
            end_time = 0.0
        else:
            start_time = (largest_run_start) * chunk_size / fps
            end_time = (largest_run_start + largest_run_length) * chunk_size / fps

        activity_intervals.append((start_time, end_time))

    return activity_intervals