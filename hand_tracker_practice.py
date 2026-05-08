import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
import time
import numpy as np
import json

base_options = python.BaseOptions(model_asset_path = "hand_landmarker.task")
HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
cap = cv2.VideoCapture(1)

mp_hands = mp.tasks.vision.HandLandmarksConnections
mp_drawing = mp.tasks.vision.drawing_utils
mp_drawing_styles = mp.tasks.vision.drawing_styles

MARGIN = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54) # vibrant green

latest_result = None
position_list = []

def draw_landmarks_on_image(rgb_image, detection_result):
    try:
        hand_landmarks_list = detection_result.hand_landmarks
        if not hand_landmarks_list:
            return rgb_image

        annotated_image = np.copy(rgb_image)
        for hand_landmarks in hand_landmarks_list:
            mp_drawing.draw_landmarks(
                annotated_image,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

        return annotated_image
    except Exception:
        return rgb_image
    
scroll_position_list = []

def scroll_down(detection_result, position_list=scroll_position_list):
    # print("HEY")
    hand_landmarks_list = detection_result.hand_landmarks
    if not hand_landmarks_list:
        return False 
    
    # x is for up and down, y is for right and left. If we go up, y coordinates decrease, if we go down, y coordinates increase. 
    for hand_landmarks in hand_landmarks_list:
        index_finger_tip = hand_landmarks[8]
        middle_finger_tip = hand_landmarks[12]

        # print("Index Finger: ", index_finger_tip.y)
        # print("Middle Finger: ", middle_finger_tip.y)
        
        position_list.append([index_finger_tip.y, middle_finger_tip.y])
        try:
            current_pos = position_list[len(position_list) - 1]
            old_pos = position_list[len(position_list) - 30]
            movement_threshold = 0.08 
            
            if (current_pos[0] - old_pos[0] > movement_threshold and 
                current_pos[1] - old_pos[1] > movement_threshold):
                print("Scrolling down")
                with open('scroll_status.json', 'w') as f:
                    json.dump({"scroll": "down"}, f)
            # elif (current_pos[0] - old_pos[0] < movement_threshold and 
            #     current_pos[1] - old_pos[1] < movement_threshold):
            #     print("Scrolling up")
        except Exception:
            continue
        
    # print(position_list)

def scroll_up(detection_result, position_list=scroll_position_list):
    # print("HEY")
    hand_landmarks_list = detection_result.hand_landmarks
    if not hand_landmarks_list:
        return False 
    
    # x is for up and down, y is for right and left. If we go up, y coordinates decrease, if we go down, y coordinates increase. 
    for hand_landmarks in hand_landmarks_list:
        index_finger_tip = hand_landmarks[8]
        middle_finger_tip = hand_landmarks[12]

        # print("Index Finger: ", index_finger_tip.y)
        # print("Middle Finger: ", middle_finger_tip.y)
        
        position_list.append([index_finger_tip.y, middle_finger_tip.y])
        try:
            current_pos = position_list[len(position_list) - 1]
            old_pos = position_list[len(position_list) - 30]
            movement_threshold = 0.08 
            
            if (current_pos[0] - old_pos[0] < movement_threshold and 
                current_pos[1] - old_pos[1] < movement_threshold):
                print("Scrolling up")
                with open('scroll_status.json', 'w') as f:
                    json.dump({"scroll": "up"}, f)
        except Exception:
            continue
        
    # print(position_list)

def detect_touch(detection_result, threshold=0.1):
    hand_landmarks_list = detection_result.hand_landmarks
    if not hand_landmarks_list:
        return False

    for hand_landmarks in hand_landmarks_list:
        try:
            thumb_tip = hand_landmarks[4]
            ring_finger_tip = hand_landmarks[16]
            pinky_tip = hand_landmarks[20]

        except Exception:
            continue

        dx_down = thumb_tip.x - ring_finger_tip.x
        dy_down = thumb_tip.y - ring_finger_tip.y
        dz_down = thumb_tip.z - ring_finger_tip.z

        dx_up = thumb_tip.x - pinky_tip.x
        dy_up = thumb_tip.y - pinky_tip.y
        dz_up = thumb_tip.z - pinky_tip.z

        if dz_down * dz_down + dy_down * dy_down + dx_down * dx_down < threshold * threshold:
            # print("YAHOO")
            # print(position_list)
            scroll_down(detection_result)
            return True
        elif dx_up * dx_up + dy_up * dy_up + dz_up * dz_up < threshold * threshold:
            scroll_up(detection_result)
            return True
    
    return False

def print_result(result: HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int): # type: ignore
    global latest_result
    latest_result = result
    position_list.append(latest_result)
    

options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.LIVE_STREAM,
    num_hands=2,
    result_callback=print_result)

detector = vision.HandLandmarker.create_from_options(options)

if not cap.isOpened():
    print("Cannot open camera")
    exit()
while True:
    ret, frame = cap.read()
    # frame = cv2.flip(frame, 1)
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    detector.detect_async(mp_image, int(time.time() * 1000))
    if latest_result is not None:
        frame = draw_landmarks_on_image(frame, latest_result)
    cv2.imshow('frame', frame)
    scrolled = detect_touch(latest_result)

    if not scrolled:
        with open('scroll_status.json', 'w') as f:
            json.dump({"scroll": "none"}, f)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()