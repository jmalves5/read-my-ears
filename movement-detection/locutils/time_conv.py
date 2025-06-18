def time_to_seconds(time_str):
    # Split the string into hours, minutes, and seconds
    h, m, s = map(float, time_str.split(':'))
    # Calculate the total seconds
    total_seconds = h * 3600 + m * 60 + s
    return total_seconds