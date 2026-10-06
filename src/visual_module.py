import cv2
import numpy as np
import onnxruntime as ort
import os
import threading
import time
from collections import deque
from utils import MODELS_DIR

CAMERA_ID = 0
POSE_MODEL_PATH = os.path.join(MODELS_DIR, "FallFusion-Pose.onnx")

KEYPOINT_NAMES = [
    "Nose", "Left Eye", "Right Eye", "Left Ear", "Right Ear",
    "Left Shoulder", "Right Shoulder", "Left Elbow", "Right Elbow",
    "Left Wrist", "Right Wrist", "Left Hip", "Right Hip",
    "Left Knee", "Right Knee", "Left Ankle", "Right Ankle"
]

camera = None
camera_lock = threading.Lock()
visual_lock = threading.Lock()
keypoints_lock = threading.Lock()

def init_camera(camera_id):
    global camera
    try:
        camera = cv2.VideoCapture(camera_id)
        if camera.isOpened():
            return camera
        camera.release()
        camera = None
    except Exception as e:
        camera = None
        raise RuntimeError("No available camera device found")

def get_frame():
    global camera, camera_lock
    with camera_lock:
        if camera is None:
            return False, None
        try:
            return camera.read()
        except Exception as e:
            print(f"Failed to get camera frame: {str(e)}")
            return False, None

def release_camera():
    global camera
    with camera_lock:
        if camera is not None:
            camera.release()
            camera = None

def visual_thread_fn(visual_result_holder, keypoints_holder, stop_event):
    try:
        providers = [
            ('CUDAExecutionProvider', {
                'device_id': 0,
                'gpu_mem_limit': 512 * 1024 * 1024,
                'arena_extend_strategy': 'kNextPowerOfTwo',
            }),
            'CPUExecutionProvider'
        ]
        session = ort.InferenceSession(POSE_MODEL_PATH, providers=providers)
    except Exception as e:
        print(f"Visual model loading failed: {str(e)}")
        session = ort.InferenceSession(POSE_MODEL_PATH, providers=['CPUExecutionProvider'])

    input_name = session.get_inputs()[0].name
    frame_times = deque(maxlen=30)
    last_frame_time = time.perf_counter()
    
    while not stop_event.is_set():
        try:
            frame_start_time = time.perf_counter()
            ret, frame = get_frame()
            if not ret:
                time.sleep(0.1)
                continue
                
            resized = cv2.resize(frame, (192, 192))
            input_data = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            input_data = np.expand_dims(input_data, axis=0).astype(np.uint8)
            
            inference_start_time = time.perf_counter()
            outputs = session.run(None, {input_name: input_data})
            inference_end_time = time.perf_counter()
            
            keypoints = outputs[0][0][0]
            with keypoints_lock:
                keypoints_holder.append((time.perf_counter(), keypoints, frame))
            score = detect_fall_score(keypoints, frame.shape[0])
            
            frame_end_time = time.perf_counter()
            frame_time = frame_end_time - frame_start_time
            frame_times.append(frame_time)
            
            current_fps = 0.0
            if len(frame_times) > 0:
                avg_frame_time = sum(frame_times) / len(frame_times)
                current_fps = 1.0 / avg_frame_time if avg_frame_time > 0 else 0.0
            
            inference_latency = inference_end_time - inference_start_time
            
            with visual_lock:
                visual_result_holder.append((time.perf_counter(), score, current_fps, inference_latency))
            
        except Exception as e:
            print(f"Error occurred during visual processing: {str(e)}")
        time.sleep(0.2)

def detect_fall_score(keypoints, frame_height, angle_threshold=30):
    CONFIDENCE_THRESHOLD = 0.15
    try:
        left_shoulder = keypoints[5]
        right_shoulder = keypoints[6]
        left_hip = keypoints[11]
        right_hip = keypoints[12]
        if min(left_shoulder[2], right_shoulder[2], left_hip[2], right_hip[2]) < CONFIDENCE_THRESHOLD:
            return 0.0
        shoulder_center = (np.array(left_shoulder[:2]) + np.array(right_shoulder[:2])) / 2
        hip_center = (np.array(left_hip[:2]) + np.array(right_hip[:2])) / 2
        dy = hip_center[0] - shoulder_center[0]
        dx = hip_center[1] - shoulder_center[1]
        angle = np.degrees(np.arctan2(abs(dy), abs(dx)))
        return 1.0 if angle < angle_threshold or angle > (180 - angle_threshold) else 0.0
    except:
        return 0.0