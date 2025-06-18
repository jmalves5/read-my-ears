import torch
import cv2
import math

def get_ear_frames(filename, frames, masked_frames, model_face, model_ear):
    if frames==[] or masked_frames==[]:
        print(frames)
        print(masked_frames)
        raise ValueError(f"some frames is empty for file {filename}")

    ear_frames = torch.tensor([])
    
    frame_height = frames[0].shape[0]
    frame_width = frames[0].shape[1]
    
    have_size = False
    for idx, frame in enumerate(frames):
        with torch.no_grad():
            results_face = model_face.predict(frame, conf=0.5, imgsz=(320), verbose=False)
            results_ear = model_ear.predict(frame, conf=0.5, imgsz=(320), verbose=False)

        # FACE
        boxes = results_face[0].boxes.cpu().numpy()  # Get bounding boxes

        # store x1,x2,y1,y2 of highest conf box
        max_conf = 0
        for idx, box_conf in enumerate(boxes.conf):
            if box_conf > max_conf:
                max_conf = box_conf
                x1, y1, x2, y2 = boxes.xyxy[idx][0], boxes.xyxy[idx][1], boxes.xyxy[idx][2], boxes.xyxy[idx][3]

        if max_conf == 0:
            if not have_size:
                continue
        

        if have_size == False:
            # get width and height of the face box
            width_face = x2 - x1
            height_face = y2 - y1
            have_size = True

        # ear box width should be half of face width
        # ear box height should be 1/6 of face height
        width_ear = int(math.floor(float(width_face) / 2))
        height_ear = int(math.floor(float(height_face) / 8.0))

        # EAR
        boxes = results_ear[0].boxes.cpu().numpy()  # Get bounding boxes
        max_conf = 0
        for idx, box_conf in enumerate(boxes.conf):
            if box_conf > max_conf:
                max_conf = box_conf
                x1, y1, x2, y2 = boxes.xyxy[idx][0], boxes.xyxy[idx][1], boxes.xyxy[idx][2], boxes.xyxy[idx][3]
                # convert to int
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    
            
        # if boxes.conf is empty, use the previous box
        if max_conf == 0:
            # use the top part of face box
            # use 2/3 of width
            x1, y1, x2, y2 = int(x1 + 0.165 * width_ear), y1, int(x2 - 0.165 * width_ear), y1 + height_ear
    
        # convert negative values to 0 and clamp to frame size
        x1 = max(0, x1)
        x2 = max(0, x2)
        y1 = max(0, y1)
        y2 = max(0, y2)

        x1 = min(frame_width, x1)
        x2 = min(frame_width, x2)
        y1 = min(frame_height, y1)
        y2 = min(frame_height, y2)

        center_ear = (int((x1+x2)/2), int((y1+y2)/2))

        # get box based on center and width_ear and length_ear
        x1 = center_ear[0] - int(width_ear)
        x2 = center_ear[0] + int(width_ear)
        y1 = center_ear[1] - int(height_ear)
        y2 = center_ear[1] + int(height_ear)

        ear_box_xyxy = [x1, y1, x2, y2]

        # convert any negative values in ear_box_xyxy to 0  
        ear_box_xyxy = [max(0, val) for val in ear_box_xyxy]

        # Extract box pixels from  frame
        try:
            ear_frame = masked_frames[idx][ear_box_xyxy[1]:ear_box_xyxy[3], ear_box_xyxy[0]:ear_box_xyxy[2]]
        except:
            print(ear_box_xyxy)
            print(idx)
            print(len(masked_frames))
            print(len(frames))

            raise ValueError(f"Could not extract ear frame for file {filename}")
        
        if ear_frame is None or ear_frame.size == 0:
            # raise error
            raise ValueError("ear frame is None")

        # get height and width of the  frame
        height, width, _ = ear_frame.shape
        
        # pad ear_frame to match aspect ratio of frame_width and frame_height
        if height > width:
            pad = int((height - width) / 2)
            ear_frame = cv2.copyMakeBorder(ear_frame, 0, 0, pad, pad, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        else:
            pad = int((width - height) / 2)
            ear_frame = cv2.copyMakeBorder(ear_frame, pad, pad, 0, 0, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        
        # resize ear_frame to match frame_width and frame_height
        ear_frame = cv2.resize(ear_frame, (frame_width, frame_height))

        # convert to tensor
        ear_frame = torch.tensor(ear_frame).permute(2, 0, 1).float()
        ear_frames = torch.cat((ear_frames, ear_frame.unsqueeze(0)), dim=0)

    return ear_frames