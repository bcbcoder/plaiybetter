import os
import cv2
from ultralytics import YOLO
import math
import csv
import numpy as np


video_path = r'file path'
video_path_out = '{}_out.avi'.format(video_path)

cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()
H, W, _ = frame.shape
frame_rate = cap.get(cv2.CAP_PROP_FPS)
frames_per_half_second = int(math.ceil(frame_rate * .3)) # how often object detection is recorded, poorly named, when increased detection happens less,
# together with check interval seconds lower means more accurate detection
model_path = os.path.join('.', 'runs', 'detect', 'train10', 'weights', 'best.pt')
model = YOLO(model_path)
threshold = 0.65

# Class names mapping
class_names = {
    0: "Lebron",
    1: "Sid the sloth",
    2: "kid laroi",
    3: "Bobby Portis",
    4: "Almond joy",
    5: "Marcus Smart"
}


trackers = {}
tracked_boxes = {}
bbox_movements = {}
prev_centers = {}
tracker_confidences = {}
check_interval_seconds = .3 # this is for the tracker and reinitializes more times as the number decreases (it is similar to
# frames_per_half_second  but instead of doing a new detection, it reinitializes tracker
frames_per_check_interval = int(frame_rate * check_interval_seconds)

output_frame_rate = frame_rate * 3  # this should increase and decrease speed of video but is not, maybe other things are effecting it
out = cv2.VideoWriter(video_path_out, cv2.VideoWriter_fourcc(*'XVID'), output_frame_rate, (W, H))

plays_data = {}
play_id = 0
frame_count = 0


def init_tracking_data():
    return {round(timestamp, 2): {class_id: {'X Coordinate': 'NA', 'Y Coordinate': 'NA', 'Direction': 'NA', 'Length': 'NA'} for class_id in class_names.keys()} for timestamp in np.arange(0, cap.get(cv2.CAP_PROP_FRAME_COUNT) / frame_rate, 1/frame_rate)}

plays_data[play_id] = init_tracking_data()

while ret:
    time_stamp = round(frame_count / frame_rate, 2)
    detected_class_ids = []
    key2 = cv2.waitKey(10)

    if frame_count % frames_per_half_second == 0 or key2 == 32:  # Check every second
        print("pressed")
        results = model(frame)[0]
        highest_score_per_class = {}

        for result in results.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = result
            if score > threshold:
                if class_id not in highest_score_per_class or score > highest_score_per_class[class_id][0]:
                    highest_score_per_class[class_id] = (score, (x1, y1, x2, y2))
                    detected_class_ids.append(class_id)

        for class_id, (score, (x1, y1, x2, y2)) in highest_score_per_class.items():
            w, h = x2 - x1, y2 - y1
            bbox = (int(x1), int(y1), int(w), int(h))

            if class_id not in trackers or frame_count % frames_per_check_interval == 0 or key2 == 32:
                print("presto")
                tracker = cv2.TrackerCSRT_create()
                tracker.init(frame, bbox)
                trackers[class_id] = tracker
                bbox_movements[class_id] = [(bbox, time_stamp)]
                tracker_confidences[class_id] = score
            tracked_boxes[class_id] = bbox

    for class_id, tracker in list(trackers.items()):
        success, box = tracker.update(frame)
        if success:
            x1, y1, w, h = [int(v) for v in box]
            tracked_boxes[class_id] = (x1, y1, w, h)
            bbox_movements[class_id].append(((x1, y1, w, h), time_stamp))
            detected_class_ids.append(class_id)
            cv2.rectangle(frame, (x1, y1), (x1 + w, y1 + h), (0, 255, 0), 2)
            class_name = class_names.get(class_id, 'Unknown Player')
            info = f"{class_name}, Box: {x1},{y1},{w},{h}, Time: {time_stamp:.2f}s"
            cv2.putText(frame, info, (x1, y1 - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            center_x = x1 + w // 2
            center_y = y1 + h // 2
            if class_id in prev_centers:
                prev_center_x, prev_center_y = prev_centers[class_id]
                HorizDir = center_x - prev_center_x
                VertDir = center_y - prev_center_y
                direction_degrees = math.degrees(math.atan2(VertDir, HorizDir)) if HorizDir or VertDir else 0
                Length = math.sqrt((HorizDir ** 2) + (VertDir ** 2))
            else:
                direction_degrees = 0
                Length = 0
            prev_centers[class_id] = (center_x, center_y)
            plays_data[play_id][time_stamp][class_id] = {
                'X Coordinate': center_x,
                'Y Coordinate': center_y,
                'Direction': direction_degrees,
                'Length': Length
            }
    out.write(frame)
    resized_frame = cv2.resize(frame, (1000, 1000))
    cv2.imshow('Video', resized_frame)
    key = cv2.waitKey(1)
    if key == 27:  
        break
    elif key == 32:  
        play_id += 1
        print("new play")
        frame_count = 0  
        plays_data[play_id] = init_tracking_data()  

    frame_count += 1
    ret, frame = cap.read()

cap.release()
out.release()
cv2.destroyAllWindows()


csv_file = 'tracked_data6.csv'
with open(csv_file, 'w', newline='') as file:
    writer = csv.writer(file)
    header = ['Play ID', 'Timestamp']
    for class_id in class_names.keys():
        header.extend([f'{class_names[class_id]} X Coordinate', f'{class_names[class_id]} Y Coordinate', f'{class_names[class_id]} Direction', f'{class_names[class_id]} Length'])
    writer.writerow(header)

    for pid, timestamps in plays_data.items():
        for time, data in timestamps.items():
            row = [pid, time]
            for class_id in class_names.keys():
                player_data = data[class_id]
                row.extend([player_data['X Coordinate'], player_data['Y Coordinate'], player_data['Direction'], player_data['Length']])
            writer.writerow(row)
