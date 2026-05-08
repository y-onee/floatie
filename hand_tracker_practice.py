import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
import time
import numpy as np

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

def sequential_processing(detection_result, position_list=scroll_position_list):
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
            if position_list[len(position_list) - 1] > position_list[len(position_list) - 4]:
                print("Scrolling down")
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

        except Exception:
            continue

        dx = thumb_tip.x - ring_finger_tip.x
        dy = thumb_tip.y - ring_finger_tip.y
        dz = getattr(thumb_tip, 'z', 0) - getattr(ring_finger_tip, 'z', 0)

        if dz * dz + dy * dy + dx * dx < threshold * threshold:
            # print("YAHOO")
            # print(position_list)
            sequential_processing(detection_result)
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
    detect_touch(latest_result)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()