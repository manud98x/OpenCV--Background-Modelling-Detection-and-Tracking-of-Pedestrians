'''
*****************************
Assignment 3 - movingObj.py 
Student ID - 7913692
*****************************
'''

import cv2 as cv
import numpy as np
import sys


## TASK 1

# Function to Classify Objects Based on the Aspect Ratio

def classify_object(stats, min_area):

    # Create an empty list to store object labels
    object_labels = []  
    object_types = {'persons': 0, 'cars': 0, 'others': 0}
    object_boxes = []  #
    
    for stat in stats:
        x, y, w, h, area = stat
        aspect_ratio = float(h) / float(w)

        # Skipping small components
        if area < min_area:  
            continue
        
        if (aspect_ratio > 1.0):
            label = 'persons'
        elif (aspect_ratio < 0.9):
            label = 'cars'
        else:
            label = 'others'
        
        # Appending the label to the list
        object_labels.append(label)  

        # Counting objects by type
        object_types[label] += 1 

        # Appending the bounding box coordinates
        object_boxes.append((x, y, x + w, y + h))  
    
    return object_labels, object_types, object_boxes


def backgroundModelling(video_path, min_area=50):

    # Creating a MOG2 background subtractor with shadow detection
    bg_subtractor = cv.createBackgroundSubtractorMOG2(detectShadows=True)

    # Opening the video capture from the specified video path
    cap = cv.VideoCapture(video_path)

    print("Processing Started...")

    # List to store object counts in each frame
    object_counts = []  

    if not cap.isOpened():
        print("Error: Could not open video capture.")
        return
    
    # Loop to continuously read frames from the video
    while cap.isOpened():  
        ret, frame = cap.read()

        if not ret:
            break

        # Resizing the frame to a standard size
        frame = cv.resize(frame, (640, 480))  

        # Subtracting the background to get the the binary foreground mask
        fg_mask = bg_subtractor.apply(frame)

        # Creating a binary mask from the foreground mask
        binary_mask = np.zeros_like(fg_mask)
        binary_mask[fg_mask > 0] = 255

        # Creating erosion and dilation kernels 
        kernel_erode = np.ones((5, 5), np.uint8)
        kernel_dilate = np.ones((7, 7), np.uint8)

        # Applying erosion and dilation to the binary mask to refine object shapes
        fg_mask = cv.erode(binary_mask, kernel_erode, iterations=2)
        fg_mask = cv.dilate(fg_mask, kernel_dilate, iterations=2)

        # Define 8-connectivity for connected component analysis
        connectivity = 8

        # Detecting connected components (objects) in the binary mask
        _, labels, stats, _ = cv.connectedComponentsWithStats(fg_mask, connectivity=connectivity)

        # Skip the background component (index 0) in statistics
        stats = stats[1:]

        # Use a function (classify_object) to obtain object labels and counts
        object_labels, object_type_counts, object_boxes = classify_object(stats, min_area=100)

        display_frame = frame.copy()  
        object_count = len(object_labels)  
        object_counts.append(object_count) 
        frame_number = int(cap.get(1))

        # Display object count and type information for the frame
        formatted_info = ", ".join([f"{v} {k}" for k, v in object_type_counts.items()])
        print(f'Frame {frame_number:04d}: {object_count} objects ({formatted_info})')

        # Converting the binary mask to a color image for visualization
        binary_mask_color = cv.cvtColor(fg_mask, cv.COLOR_GRAY2BGR)

        # Erode and dilate the binary mask again 
        binary_mask = cv.erode(binary_mask, kernel_erode, iterations=2)
        binary_mask = cv.dilate(binary_mask, kernel_erode, iterations=2)
        
        # Applying the binary mask to the original frame to extract objects
        colour_objects = cv.bitwise_and(display_frame, display_frame, mask=binary_mask)

        # Creating the first row with two frames side by side
        frame1 = cv.resize(frame, (640, 360))  
        frame2 = cv.resize(bg_subtractor.getBackgroundImage(), (640, 360)) 
        row1 = np.hstack((frame1, frame2))

        # Creating the second row with two frames side by side
        frame3 = cv.resize(binary_mask_color, (640, 360))  
        frame4 = cv.resize(colour_objects, (640, 360)) 
        row2 = np.hstack((frame3, frame4))

        # Stacking the two rows vertically 
        combined_frame = np.vstack((row1, row2))

        cv.imshow('Background Modeling', combined_frame)

        # Checking for the 'Esc' key
        if cv.waitKey(30) & 0xFF == 27:
            break

    # Release the video capture and close any open windows
    cap.release()
    cv.destroyAllWindows()



## TASK 2

'''
Description
-----------

The solution written in Python provides a solution for pedestrian recognition and tracking in video streams. 
It recognizes pedestrians with confidence scores using a pre-trained MobileNet SSD model, and it tracks objects using OpenCV's KCF trackers. 
The main technique is based on closeness in location to match observed pedestrians to existing trackers, 
assuming that pedestrians close to each other in consecutive frames are the same persons. For matching,
the camera is assumed to be stationary at the bottom left corner of the scene, and a constant distance threshold is utilized. 
Each pedestrian is given a distinct tracking number. Using the calculate_distance() funcation the distance to pedestrians are generated.
After the pedestrian distances are sorted and obtained the 3 most closest to (0,0) camere location


'''


# Function to Calculate distance o pedestrian from the Cameta
def calculate_distance(bbox, camera_location=(0, 0)):
    x, y, _, _ = bbox
    cx, cy = camera_location
    distance = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    return distance

# Fuction to detect and track pedestrians
def detect_and_track(video_path):


    # Loading a pre-trained model 
    net = cv.dnn.readNet('frozen_inference_graph.pb', 'ssd_mobilenet_v2_coco_2018_03_29.pbtxt.txt', framework='TensorFlow')

    # Reading the classes used in the object detection model
    with open('object_detection_classes_coco.txt', 'r') as f:
        classes = f.read().strip().split('\n')

    # Initializing video capture from a specified video source (e.g., a file or camera)
    cap = cv.VideoCapture(video_path)

    # Checking if the video capture is successful;
    if not cap.isOpened():
        print("Error: Could not open video capture.")
        return

    # Empty set to keep track of assigned pedestrian numbers
    assigned_numbers = set()

    # Dictionaries to store trackers and tracked pedestrian positions
    trackers = {}  #
    tracked_pedestrians = {}  

    # Loop for processing each frame in the video
    while True:
        ret, frame = cap.read()

        # Check if reading the frame was successful
        if not ret:
            break

        # Creating a copy of the frame for image processing
        image = frame.copy()
        image_height, image_width, _ = image.shape

        # Preparing the frame for pedestrian detection by creating a blob
        blob = cv.dnn.blobFromImage(image, size=(300, 300), mean=(104, 117, 123), swapRB=True)
        net.setInput(blob)
        output = net.forward()

        # Creating copies of the frame for visualization
        frame2 = frame.copy()
        frame3 = frame.copy()
        frame4 = frame.copy()

        # Creating lists to store detected and tracked pedestrians
        detected_pedestrians = []
        tracked_closest_pedestrians = []

        # Creating a new list for detected pedestrians (clearing the previous one)
        detected_pedestrians = []

        # Creating an empty list for closest pedestrians
        closest_pedestrians = []

        # Loop through each detection in the output
        for detection in output[0, 0, :, :]:
            confidence = detection[2]

            # Check if the detection has sufficient confidence
            if confidence > 0.5:
                class_id = int(detection[1])
                class_name = classes[class_id - 1]

                # Check if the detected object is a person
                if class_name == 'person':
                    box_x = int(detection[3] * image_width)
                    box_y = int(detection[4] * image_height)
                    box_width = int(detection[5] * image_width) - box_x
                    box_height = int(detection[6] * image_height) - box_y

                    # Create a bounding box (bbox) for the detected pedestrian
                    bbox = (box_x, box_y, box_width, box_height)
                    detected_pedestrians.append(bbox)

        # Sort the detected pedestrians based on their distance from the camera location (0,0)
        detected_pedestrians.sort(key=lambda bbox: calculate_distance(bbox, (0, 0)))

        # Iterate over the top 3 detected pedestrians
        for i, bbox in enumerate(detected_pedestrians[:3]):
            matched = False

            # Search for a matching tracker based on the detection coordinates
            for number, last_known_position in tracked_pedestrians.items():
                x, y, w, h = last_known_position
                last_bbox = (x, y, w, h)

                # Check if the detected object matches the existing tracker
                if calculate_distance(bbox, (x, y)) < 200:  # Threshold for matching existing trackers
                    matched = True
                    # Update the tracker with the current frame
                    trackers[number].update(frame)
                    # Update the last known position of the pedestrian
                    tracked_pedestrians[number] = bbox
                    break

            # If no match was found for the detected pedestrian
            if not matched:
                # Assign a new unique number to the pedestrian
                unique_number = 1
                while unique_number in assigned_numbers:
                    unique_number += 1
                assigned_numbers.add(unique_number)
                # Create a new tracker for the pedestrian
                tracker = cv.TrackerKCF_create()
                tracker.init(frame, bbox)
                trackers[unique_number] = tracker
                tracked_pedestrians[unique_number] = bbox

        # Iterate through tracked pedestrians and update their positions
        for number, last_known_position in tracked_pedestrians.items():
            x, y, w, h = last_known_position
            tracker = trackers[number]
            success, tracked_bbox = tracker.update(frame)
            x, y, w, h = [int(i) for i in tracked_bbox]
            cv.rectangle(frame2, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Draw a green bounding box
            cv.putText(frame2, 'Person', (x, y - 15), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Iterate through tracked pedestrians and update their positions for visualization
        for number, last_known_position in tracked_pedestrians.items():
            x, y, w, h = last_known_position
            tracker = trackers[number]
            success, tracked_bbox = tracker.update(frame)
            x, y, w, h = [int(i) for i in tracked_bbox]
            cv.rectangle(frame3, (x, y), (x + w, y + h), (0,0, 255), 2)  # Draw a green bounding box
            cv.putText(frame3, f'Person {number}', (x, y - 15), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Create a dictionary to store the distances of tracked pedestrians from the camera's bottom left corner
        distances = {}

        # Calculate the distance of each tracked pedestrian
        for number, last_known_position in tracked_pedestrians.items():
            x, y, w, h = last_known_position
            distance = calculate_distance((x, y + h, w, h), (0, image_height))
            distances[number] = distance

        # Sort the pedestrians by their distance from the camera's bottom left corner
        sorted_pedestrians = sorted(distances.items(), key=lambda item: item[1])

        # Get the top 3 closest pedestrians
        closest_pedestrians_frame3 = sorted_pedestrians[:3]

        # Draw blue bounding boxes for the closest pedestrians
        for i, (number, _) in enumerate(closest_pedestrians_frame3):
            x, y, w, h = tracked_pedestrians[number]
            tracker = trackers[number]
            success, tracked_bbox = tracker.update(frame)
            x, y, w, h = [int(i) for i in tracked_bbox]
            cv.rectangle(frame4, (x, y), (x + w, y + h), (255, 0, 0), 2)  # Draw a blue bounding box
            cv.putText(frame4, 'Closest Out of Detected', (x, y - 15), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        # Resize frames for display
        frame = cv.resize(frame, (640, 360))
        detected_frame = cv.resize(frame2, (640, 360))
        tracked_frame = cv.resize(frame3, (640, 360))
        closest_ped_frame = cv.resize(frame4, (640, 360))

        # Combine the frames into a single visualization
        row1 = np.hstack((frame, detected_frame))
        row2 = np.hstack((tracked_frame, closest_ped_frame))
        combined = np.vstack((row1, row2))

        # Display the combined frame with detected and tracked pedestrians
        cv.imshow('Detection and Tracking of Pedestrians', combined)

        # Check for the 'Esc' key (27) to exit the loop
        if cv.waitKey(30) & 0xFF == 27:
            break

    # Release the video capture and close any open windows
    cap.release()
    cv.destroyAllWindows()



def main():
    if len(sys.argv) != 3:
        print("Usage: python movingObj.py <-b|-d> <video_file>")
        return
    option = sys.argv[1]
    video_file = sys.argv[2]

    if option == "-b":
        backgroundModelling(video_file)
    elif option == "-d":
        detect_and_track(video_file)
    else:
        print("Invalid option. Use '-b' for Task One or '-d' for Task Two.")

if __name__ == "__main__":
    main()
        
