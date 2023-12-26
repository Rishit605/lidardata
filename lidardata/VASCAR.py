import csv
import cv2
import torch
from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO('yolov8m.pt')

# Open the video file
video_path = "test2.mp4"
cap = cv2.VideoCapture(video_path)


def vid_param():
    # Define the distance between two landmarks in meters
    DISTANCE = 0.5

    # Define the focal length of the camera in mm
    FOCAL_LENGTH = 12

    # Define the width of the camera sensor in mm
    SENSOR_WIDTH = 6.17

    # Define the width of the video frame in pixels
    PIXEL_WIDTH = 640
    return DISTANCE, FOCAL_LENGTH, SENSOR_WIDTH, PIXEL_WIDTH


# Initialize dictionaries for vehicle data and speed history
vehicles = {}
speed_history = {}
dist = []

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Run YOLOv8 tracking on the frame, persisting tracks between frames
        results = model.track(frame, persist=True, classes=2, device=0)

        # Visualize the results on the frame
        annotated_frame = results[0].plot()

        # For each tracked object, do the following:
        for obj in results[0]:
            try:
                # Get bounding box coordinates, class label, and ID (with proper handling for None)
                x1, y1, x2, y2 = obj.boxes[0].xyxy[0].cpu().numpy() if obj.boxes[0] is not None else (0, 0, 0, 0)
                label = obj.boxes[0].cls if obj.boxes[0] is not None else None
                ID = tuple(obj.boxes[0].id.cpu().numpy()) if obj.boxes[0] and obj.boxes[0].id is not None else None

                # Calculate midpoint and current distance (with handling for None)
                x = (x1 + x2) / 2 if x1 != 0 and x2 != 0 else 0
                y = (y1 + y2) / 2 if y1 != 0 and y2 != 0 else 0
                # curr_distance = (FOCAL_LENGTH * SENSOR_WIDTH * DISTANCE) / (x * PIXEL_WIDTH) if x != 0 else None
                curr_distance = (vid_param()[1] * vid_param()[2] * vid_param()[0]) / (x * vid_param()[3]) if x != 0 else None

                # Check if vehicle ID is in the dictionary and handle None values
                if ID in vehicles:
                    prev_distance, prev_time = vehicles[ID]
                    print(prev_distance)
                    curr_time = cv2.getTickCount() / cv2.getTickFrequency()

                    # Check if vehicle has crossed second landmark with proper None handling
                    if curr_distance is not None and prev_distance is not None:
                        if curr_distance < (prev_distance - vid_param()[0]) and curr_distance is not None:

                            # if curr_distance > 0 and curr_distance is not None: # For Later Trail and Testing
                            prev_distance, prev_time = vehicles.get(ID, (0, 0))
                            speed = vid_param()[0] / (curr_time - prev_time)
                            vehicles[ID] = (curr_distance, curr_time)
                            # print("SPEED: ", speed)

                            # Draw the speed on the frame
                            cv2.putText(annotated_frame, f'Speed: {speed:.2f} m/s', (int(x1), int(y1) - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))

                            if ID in speed_history:
                                speed_history[ID].append(speed)
                            else:
                                speed_history[ID] = [speed]



                        else:
                            vehicles[ID] = (curr_distance * 1000, curr_time)
                    else:
                        pass
                # Update vehicle data if not crossed second landmark (handle None)
                else:
                    if curr_distance is not None:
                        vehicles[ID] = ((vid_param()[1] * vid_param()[2] * vid_param()[0]) / (x * vid_param()[3]),
                                        cv2.getTickCount() / cv2.getTickFrequency())

            except AttributeError:
                # Handle AttributeError raised by accessing attributes on None objects
                pass

        # Display the annotated frame
        cv2.imshow("Frame", annotated_frame)

        # Break loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        break

# Release resources
cap.release()
cv2.destroyAllWindows()

