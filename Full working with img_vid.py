import cv2
import torch  # For using the PyTorch model
import numpy as np


# Load the pre-trained YOLO model (PyTorch)
model = torch.hub.load('C:/Users/ASUS/Documents/NOTES/Minor project/minor project/pythonProject/yolov7',
                       'custom',
                       'C:/Users/ASUS/Documents/NOTES/Minor project/minor project/pythonProject/yolov7/runs/train/'
                       'exp/weights/best.pt', source='local')

model.eval()  # Set model to evaluation mode

# Function to apply Non-Maximum Suppression (NMS)
def apply_nms(detections, iou_threshold=0.4):
    if len(detections) == 0:
        return []

    #boxes = np.array([[x1, y1, x2, y2] for x1, y1, x2, y2, _, _ in detections])
    boxes = np.array([[det[0], det[1], det[2], det[3]] for det in detections])
    #scores = np.array([conf for _, _, _, _, _, conf in detections])
    scores = np.array([conf for _, _, _, _, _, conf, *_ in detections])

    indices = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), score_threshold=0.5, nms_threshold=iou_threshold)
    indices = indices.flatten() if len(indices) > 0 else []
    return [detections[i] for i in indices]

# Function to process a single image with tiling for large images
def process_image(image_path):
    # Load the image
    frame = cv2.imread(image_path)
    if frame is None:
        print("Error: Unable to load image.")
        return

    height, width, _ = frame.shape
    tile_size = 640  # Define the size of each tile
    overlap = 100  # Define overlap between tiles

    results_list = []

    # Loop over the image in tiles
    for y in range(0, height, tile_size - overlap):
        for x in range(0, width, tile_size - overlap):
            # Crop the tile
            x_end = min(x + tile_size, width)
            y_end = min(y + tile_size, height)
            tile = frame[y:y_end, x:x_end]

            # Convert tile to RGB for the model
            img = cv2.cvtColor(tile, cv2.COLOR_BGR2RGB)

            # Perform detection
            results = model(img)

            # Parse results and store detections
            for *xyxy, conf, cls in results.xyxy[0]:
                label = model.names[int(cls)]
                confidence = float(conf)

                if label in ["ambulance", "police", "firetruck"] and confidence > 0.5:
                    x1, y1, x2, y2 = map(int, xyxy)
                    # Adjust coordinates to the original image
                    results_list.append((x1 + x, y1 + y, x2 + x, y2 + y, label, confidence))

    # Apply NMS to remove duplicate bounding boxes
    results_list = apply_nms(results_list)

    # Draw all detections on the original image
    for x1, y1, x2, y2, label, confidence in results_list:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"{label} ({confidence * 100:.2f}%)", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 255, 0), 2)

    # Display the processed image
    cv2.imshow("Emergency Vehicle Detection", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Function to process a recorded video
def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Unable to open video.")
        return

    frame_count = 0
    detections = []  # Store detections for persistence
    max_persistence = 10  # Number of frames to persist detections

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Resize frame for faster processing
        frame = cv2.resize(frame, (640, 480))  # Adjust dimensions as needed

        if frame_count % 4 == 0:
            # Convert frame to RGB (PyTorch models expect RGB input)
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Perform detection with the model
            results = model(img)

            # Parse the results and store bounding boxes
            current_detections = []
            for *xyxy, conf, cls in results.xyxy[0]:  # xyxy are the bounding box coordinates
                label = model.names[int(cls)]
                confidence = float(conf)

                if label in ["ambulance", "police", "firetruck"] and confidence > 0.5:
                    x1, y1, x2, y2 = map(int, xyxy)
                    current_detections.append((x1, y1, x2, y2, label, confidence, frame_count))

            # Apply NMS to current detections
            current_detections = apply_nms(current_detections)

            # Add current detections to the list
            detections.extend(current_detections)

        # Draw existing detections, and remove stale ones
        valid_detections = []
        for x1, y1, x2, y2, label, confidence, detected_frame in detections:
            if frame_count - detected_frame <= max_persistence:
                valid_detections.append((x1, y1, x2, y2, label, confidence, detected_frame))
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{label} ({confidence * 100:.2f}%)", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (0, 255, 0), 2)
        detections = valid_detections

        # Display the video feed with bounding boxes and labels
        cv2.imshow("Emergency Vehicle Detection", frame)

        # Exit if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_count += 1

    cap.release()
    cv2.destroyAllWindows()

# Example usage:
process_image("C:/Users/ASUS/Documents/NOTES/Minor project/minor project/imgPATH/5.png")
process_video("C:/Users/ASUS/Documents/NOTES/Minor project/minor project/imgPATH/3.mp4")
