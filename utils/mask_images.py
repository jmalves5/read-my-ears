import numpy as np
import glob
import cv2
from tqdm import tqdm

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')

# Read images from dir into arrays
# Loop over arrays, extract pictures and create masked version
# Outside the loop create mp4 file with masked versions

frames_path = "/home/joao/workspace/EquinePainFaceDataset/dataset/clips/25FPS/original/frames"

# loop over folders inside frames path
for folder in tqdm(glob.glob(f"{frames_path}/*")):
    # remove _frames from end of folder name
    folder_no_frames = folder.split("_frames")[0]
    # get folder name
    folder_no_frames = folder_no_frames.split("/")[-1]
    out = cv2.VideoWriter(f"/home/joao/workspace/EquinePainFaceDataset/dataset/clips/25FPS/masked/{folder_no_frames}_.mp4", fourcc, 25.0, (1920, 1080))
    # start index shoiuld be 00000
    frames_cap = cv2.VideoCapture(f"{folder}/%5d.jpg")
    # start index should be frame_1.png
    masks_cap = cv2.VideoCapture(f"{folder}/masks/frame_%d.png")
    frame_counter = 0
    while(True):
        orig_ret, orig_frame = frames_cap.read()
        mask_ret, mask_frame = masks_cap.read()
        if orig_ret == True and mask_ret == True:
            frame_counter += 1

            if frame_counter == frames_cap.get(cv2.CAP_PROP_FRAME_COUNT):
                break
            
            result = cv2.bitwise_and(orig_frame, mask_frame)

            # Mask input image with binary mask
            # Color background black
            result[mask_frame==0] = 0
            
            # Write to 
            if result is not None:
                print(f"Writing frame {frame_counter}")
                out.write(result)

            if cv2.waitKey(1) & 0xFF == ord('s'): 
                break

        # Break the loop 
        else: 
            break
    frames_cap.release()
    masks_cap.release() 
    out.release() 
    
# Closes all the frames 
cv2.destroyAllWindows() 
