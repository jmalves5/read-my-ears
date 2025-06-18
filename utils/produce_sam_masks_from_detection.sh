#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh

# Run horse-face-detection on data
for fps in 25; do

    export video_frames_path="/home/joao/workspace/EquinePainFaceDataset/dataset/clips/25FPS/original/frames/"

    # iterate over all videos in path
    for frames_path in "/home/joao/workspace/EquinePainFaceDataset/dataset/clips/25FPS/original/frames/action_S9.mp4_10_frames"; do
        cd extract-ear-frames
        cd horse_face_detection
        conda activate base
        conda activate horse-face-detection

        export box=$(YOLO_VERBOSE=False python3 infer_first.py $frames_path)

        cd ../..

        echo Running SAM2
        # Run SAM2 on the images given the initial bbox
        cd sam2
        conda activate base
        conda activate sam2
        echo $frames_path
        echo $box
        python3 inference.py $frames_path $box

        cd ..
        conda deactivate
    done
done


