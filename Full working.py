import cv2
import torch  # For using the PyTorch model
import serial  # For communication with Arduino
import time

# Load the pre-trained YOLO model (PyTorch)
model = torch.hub.load('C:/Users/ASUS/Documents/NOTES/Minor project/minor project/pythonProject/yolov7',
                       'custom',
                       'C:/Users/ASUS/Documents/NOTES/Minor project/minor project/pythonProject/yolov7/runs/train/'
                       'exp/weights/best.pt', source='local')

model.eval()  # Set model to evaluation mode
# Initialize serial communication with Arduino
arduino = serial.Serial('COM3', 9600)
time.sleep(2)  # Allow time for Arduino to reset after establishing the connection

# Testing phase: Send a signal to Arduino to test all lights and the buzzer
print("Initiating testing phase...")
arduino.write(b'T')  # Send 'T' to Arduino to start the testing phase
time.sleep(5)  # Wait for 5 seconds to allow Arduino to complete the testing phase
print("Testing phase completed. Starting detection...")

# Initialize video capture (0 for webcam)
cap = cv2.VideoCapture(0)
frame_count = 0

while True:
    # Capture frame-by-frame from the webcam
    ret, frame = cap.read()
    if not ret:
        break  # If the frame is not grabbed, break out of the loop

    # Resize frame for faster processing
    frame = cv2.resize(frame, (640, 480))  # Adjust dimensions as needed

    # Convert frame to RGB (PyTorch models expect RGB input)
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    if frame_count % 1 == 0:  # Process every frame
        # Perform detection with the model
        results = model(img)

        # Flag to check if an emergency vehicle is detected
        emergency_detected = False

        # Parse the results and draw bounding boxes
        for *xyxy, conf, cls in results.xyxy[0]:  # xyxy are the bounding box coordinates
            # Get the class name from the model's labels
            label = model.names[int(cls)]
            confidence = float(conf)

            # Check if it's an emergency vehicle (adjust based on your trained classes)
            if label in ["ambulance", "police", "firetruck"] and confidence > 0.5:
                emergency_detected = True  # Set flag to True if any emergency vehicle is detected

                # Draw bounding box and label on the frame
                x1, y1, x2, y2 = map(int, xyxy)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{label} ({confidence * 100:.2f}%)", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (0, 255, 0), 2)

        # If an emergency vehicle is detected, send a signal to the Arduino
        if emergency_detected:
            arduino.write(b'1')  # Send '1' to Arduino for emergency detected
            print("Emergency vehicle detected. Signal sent to Arduino.")
        else:
            arduino.write(b'0')  # Send '0' to Arduino for no emergency
            print("No emergency detected. Normal operation.")

    # Display the video feed with bounding boxes and labels
    cv2.imshow("Emergency Vehicle Detection", frame)

    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frame_count += 1

# Release the video capture and close the OpenCV windows
cap.release()
cv2.destroyAllWindows()
arduino.close()  # Close the serial connection
