import cv2
import mediapipe as mp
import math
from concurrent.futures import ThreadPoolExecutor

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=True, min_detection_confidence=0.3, model_complexity=2
)
mp_drawing = mp.solutions.drawing_utils

def calculateAngle(landmark1, landmark2, landmark3):
    x1, y1, _ = landmark1
    x2, y2, _ = landmark2
    x3, y3, _ = landmark3

    angle = math.degrees(math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2))
    
    if angle < 0:
        angle += 360
    
    return angle

def classifyPose(landmarks):
    label = 'Unknown Pose'
    angles = {
        'left_elbow': calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                                      landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value],
                                      landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]),
        'right_elbow': calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
                                       landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value],
                                       landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]),
        'left_shoulder': calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value],
                                         landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                                         landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]),
        'right_shoulder': calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],
                                          landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
                                          landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value]),
        'left_knee': calculateAngle(landmarks[mp_pose.PoseLandmark.LEFT_HIP.value],
                                     landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value],
                                     landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value]),
        'right_knee': calculateAngle(landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],
                                      landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value],
                                      landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value])
    }

    left_elbow_angle = angles['left_elbow']
    right_elbow_angle = angles['right_elbow']
    left_shoulder_angle = angles['left_shoulder']
    right_shoulder_angle = angles['right_shoulder']
    left_knee_angle = angles['left_knee']
    right_knee_angle = angles['right_knee']
    
    if landmarks[19][1] > landmarks[23][1] and landmarks[20][1] > landmarks[24][1]:
        if left_elbow_angle > 170 and right_elbow_angle > 170 and left_shoulder_angle > 200 and right_shoulder_angle < 120:
            if left_knee_angle > 160 or right_knee_angle > 160:
                label = 'Plank'
            elif left_knee_angle < 160 and right_knee_angle < 160:
                label = 'Knee Plank'

        if left_elbow_angle < 120 and right_elbow_angle < 120 and left_shoulder_angle < 120 and right_shoulder_angle > 200:
            if left_knee_angle < 200 or right_knee_angle < 200:
                label = 'Plank'
            elif left_knee_angle > 200 and right_knee_angle > 200:
                label = 'Knee Plank'
    
    if landmarks[19][1] < landmarks[23][1] and landmarks[20][1] < landmarks[24][1] and landmarks[25][1] > landmarks[0][1] and landmarks[26][1] > landmarks[0][1]:
        if landmarks[23][1] > landmarks[11][1] and landmarks[24][1] > landmarks[12][1]:
            if left_elbow_angle > 130 and right_elbow_angle > 130 and left_shoulder_angle > 200 or right_shoulder_angle < 120:
                label = 'Squat'
            if left_elbow_angle < 120 and right_elbow_angle < 120 and left_shoulder_angle < 120 or right_shoulder_angle > 200:
                label = 'Squat'
                
    if landmarks[23][1] > landmarks[25][1] and landmarks[24][1] > landmarks[26][1]:
        if landmarks[13][1] < landmarks[0][1] and landmarks[14][1] < landmarks[0][1] or landmarks[13][1] < landmarks[23][1] and landmarks[13][1] > landmarks[0][1] and landmarks[14][1] < landmarks[24][1] and landmarks[14][1] > landmarks[0][1]:
            if landmarks[29][0] < landmarks[25][0] and landmarks[25][0] < landmarks[11][0] or landmarks[30][0] < landmarks[26][0] and landmarks[26][0] < landmarks[12][0]:
                calculateFootRise = (landmarks[23][1] + landmarks[25][1])/2 > landmarks[29][1]
                calculateFootRise1 = (landmarks[24][1] + landmarks[26][1])/2 > landmarks[30][1]
                if left_elbow_angle > 130 and right_elbow_angle > 130 and left_shoulder_angle > 200 or right_shoulder_angle < 130:
                    label = "Situp"
                    if calculateFootRise and calculateFootRise1:
                        label = 'foot rise Situp'

                if left_elbow_angle < 130 and right_elbow_angle < 130 and left_shoulder_angle < 130 or right_shoulder_angle > 200:
                    label = 'Situp'
                    if calculateFootRise and calculateFootRise1:
                        label = 'foot rise Situp'

    return label

def apply_gaussian_blur(image):
    return cv2.GaussianBlur(image, (5, 5), 0)

def process_frame(image, pose, type, answer):
    sample_img = apply_gaussian_blur(image)
    results = pose.process(cv2.cvtColor(sample_img, cv2.COLOR_BGR2RGB))
    imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(imageRGB)
    height, width, _ = image.shape

    if results.pose_landmarks:
        mp_drawing.draw_landmarks(image=sample_img, landmark_list=results.pose_landmarks, connections=mp_pose.POSE_CONNECTIONS)

    landmarks = []
    if results.pose_landmarks:
        for landmark in results.pose_landmarks.landmark:
            landmarks.append((int(landmark.x * width), int(landmark.y * height), (landmark.z * width)))
        result = classifyPose(landmarks)
        
        for i, type1 in enumerate(type):
            if result == type1:
                answer[i] = answer[i] + 1
    return answer

def video_parallel(PATH):
    vidcap = cv2.VideoCapture(PATH)
    type = ["Plank", "Knee Plank", "Squat", "Situp", "foot rise Situp", "Unknown Pose"]
    answer = [0] * len(type)
    
    with ThreadPoolExecutor() as executor:
        futures = []
        while vidcap.isOpened():
            ret, image = vidcap.read()
            if ret == False:
                break
            if int(vidcap.get(1)) % 60 == 0:
                future = executor.submit(process_frame, image, pose, type, answer)
                futures.append(future)

        for future in futures:
            answer = future.result()
                
    vidcap.release()
    return type[answer.index(max(answer))]