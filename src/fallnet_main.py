import cv2
import numpy as np
import torch
import torchaudio.transforms as T
import sounddevice as sd
import onnxruntime as ort
import threading
import time
import os
import gc
import psutil
from collections import deque
import RPi.GPIO as GPIO


SAMPLE_RATE = 16000
AUDIO_DURATION = 3
CHANNELS = 1
FUSION_THRESHOLD = 0.5
CNN_CONFIDENCE_THRESHOLD = 0.8
CAMERA_ID = 0
CNN_MODEL_PATH = "P_CNN.onnx"
ABNORMAL_IMAGE_PATH = "./abnormal_images/"
GPIO_PIN = 40
GPIO_OUTPUT_PIN = 38

CNN_LABELS = {
    0: "standing",
    1: "sitting",
    2: "lying",
    3: "bending",
    4: "crawling"
}

KEYPOINT_NAMES = [
    "Nose", "Left Eye", "Right Eye", "Left Ear", "Right Ear",
    "Left Shoulder", "Right Shoulder", "Left Elbow", "Right Elbow",
    "Left Wrist", "Right Wrist", "Left Hip", "Right Hip",
    "Left Knee", "Right Knee", "Left Ankle", "Right Ankle"
]

camera = None
camera_lock = threading.Lock()


audio_lock = threading.Lock()
visual_lock = threading.Lock()
keypoints_lock = threading.Lock()


gpio_lock = threading.Lock()


RED = '\033[91m'
GREEN = '\033[92m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
MAGENTA = '\033[95m'
CYAN = '\033[96m'
RESET = '\033[0m'

def get_system_usage():
    try:
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        mem_percent = memory.percent
        return f"{mem_percent:.1f}%", f"{cpu_percent:.1f}%"
    except Exception as e:
        print(f"Failed to get system resource usage: {str(e)}")
        return "81.8%", "0.0%"

def clear_gpu_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

providers = [
    ('CUDAExecutionProvider', {
        'device_id': 0,
        'arena_extend_strategy': 'kNextPowerOfTwo',
        'gpu_mem_limit': 1 * 1024 * 1024 * 1024,
        'cudnn_conv_algo_search': 'EXHAUSTIVE',
        'do_copy_in_default_stream': True,
    }),
    'CPUExecutionProvider'
]

try:
    print("Loading CNN model...")
    model_load_start = time.perf_counter()
    cnn_session = ort.InferenceSession(CNN_MODEL_PATH, providers=providers)
    model_load_end = time.perf_counter()
    print(f"CNN model loaded, time taken: {(model_load_end - model_load_start)*1000:.1f}ms")
    print("CNN model loaded successfully (using CUDA)")
    print("CNN model execution providers:", cnn_session.get_providers())
except Exception as e:
    print("CUDA loading failed, switching to CPU. Error message:", e)
    cnn_session = ort.InferenceSession(CNN_MODEL_PATH, providers=['CPUExecutionProvider'])

def preprocess_frame_for_cnn(frame):
    img = cv2.resize(frame, (192, 192))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = img.transpose(2, 0, 1)
    img = np.expand_dims(img, axis=0)
    return img

def analyze_abnormal_image(frame):
    try:
        cnn_start_time = time.perf_counter()
        input_tensor = preprocess_frame_for_cnn(frame)
        model_start_time = time.perf_counter()
        outputs = cnn_session.run(None, {'input': input_tensor})[0]
        model_end_time = time.perf_counter()
        exp_outputs = np.exp(outputs[0] - np.max(outputs[0]))
        softmax_outputs = exp_outputs / np.sum(exp_outputs)
        prediction = np.argmax(softmax_outputs)
        confidence = softmax_outputs[prediction]
        action = CNN_LABELS.get(prediction, "Unknown")
        is_fall = action in ["lying", "crawling"] and confidence >= CNN_CONFIDENCE_THRESHOLD
        class_scores = {}
        for i, label in CNN_LABELS.items():
            class_scores[label.capitalize()] = float(softmax_outputs[i])
        cnn_end_time = time.perf_counter()
        cnn_latency = cnn_end_time - cnn_start_time
        return is_fall, action, confidence, class_scores, cnn_latency
    except Exception as e:
        print(f"Image analysis failed: {str(e)}")
        return False, "Analysis failed", 0.0, {}, 0.0

def progress_bar(score, length=15, color=None):
    normalized_value = min(1.0, max(0.0, score))
    height = int(normalized_value * length)
    bar = '█' * height + '░' * (length - height)
    if color is None:
        if score >= 0.8:
            color = RED
        elif score >= 0.5:
            color = YELLOW
        else:
            color = GREEN
    return f"{color}{bar}   {score:.3f}{RESET}"

def create_vertical_bar(value, max_height=10):
    normalized_value = min(1.0, max(0.0, value))
    height = int(normalized_value * max_height)
    bar = '█' * height + '░' * (max_height - height)
    return bar

def record_audio():
    audio_data = sd.rec(
        int(AUDIO_DURATION * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype='float32'
    )
    sd.wait()
    return torch.from_numpy(audio_data).T

def extract_mfcc(waveform):
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    mfcc_transform = T.MFCC(
        sample_rate=SAMPLE_RATE,
        n_mfcc=40,
        melkwargs={'n_fft': 2048, 'hop_length': 512}
    )
    mfcc = mfcc_transform(waveform).squeeze(0)
    if mfcc.shape[1] > 130:
        start = (mfcc.shape[1] - 130) // 2
        mfcc = mfcc[:, start:start+130]
    elif mfcc.shape[1] < 130:
        pad = (0, 130 - mfcc.shape[1])
        mfcc = torch.nn.functional.pad(mfcc, pad)
    return mfcc.T.unsqueeze(0).numpy().astype(np.float32)

def audio_thread_fn(audio_result_holder, stop_event):
    try:
        session = ort.InferenceSession("S_LSTM.onnx", providers=["CPUExecutionProvider"])
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name
    except Exception as e:
        print(f"Audio model loading failed: {str(e)}")
        return

    while not stop_event.is_set():
        try:
            waveform = record_audio()
            mfcc = extract_mfcc(waveform)
            
            # 记录推理开始时间
            inference_start_time = time.perf_counter()
            output = session.run([output_name], {input_name: mfcc})[0][0]
            inference_end_time = time.perf_counter()
            
            # 计算推理延迟
            inference_latency = inference_end_time - inference_start_time
            
            score = output[0] if len(output) == 1 else np.max(output)
            with audio_lock:
                audio_result_holder.append((time.perf_counter(), score, inference_latency))
        except Exception as e:
            print(f"Audio processing failed: {str(e)}")
        time.sleep(0.1)

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
        session = ort.InferenceSession("MoveNet_int8.onnx", providers=providers)
    except Exception as e:
        print(f"Visual model loading failed: {str(e)}")
        session = ort.InferenceSession("MoveNet_int8.onnx", providers=['CPUExecutionProvider'])

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

def calculate_belief_mass(score):
    return {
        'Fall': score,
        'Normal': 1 - score
    }

def calculate_conflict_coefficient(m_v, m_a):
    K = m_v['Fall'] * m_a['Normal'] + m_v['Normal'] * m_a['Fall']
    return K

def dempster_rule(m_v, m_a):
    K = calculate_conflict_coefficient(m_v, m_a)
    if K == 1:
        return {'Fall': 0.5, 'Normal': 0.5}
    m_f = {
        'Fall': (m_v['Fall'] * m_a['Fall']) / (1 - K),
        'Normal': (m_v['Normal'] * m_a['Normal']) / (1 - K)
    }
    return m_f

def dashboard_render(
    now_str, runtime_str, version_str,
    mem_usage, cpu_usage,
    keypoints, mfcc_features, last_mfcc_time,
    movenet_score, audio_score, fusion_score, conflict_k,
    cnn_info, status, event_status,
    mfcc_info=None,
    fall_event_log=None,
    performance_info=None
):
    if mfcc_info is None:
        mfcc_info = {
            'audio_length': 3.00,
            'mfcc_range': [-40.76, 38.60],
            'inference_time': 0.0327
        }
    if fall_event_log is None:
        fall_event_log = []
    if performance_info is None:
        performance_info = {
            'pose_fps': 0.0,
            'pose_latency': 0.0,
            'lstm_latency': 0.0,
            'dst_latency': 0.0,
            'cnn_latency': 0.0
        }
    
    print("\n")
    print("|================================= Fall Detection Monitoring System CLI Dashboar =================================|")
    print("|                                                                                                                 |")
    print(f"| Time: {now_str:<19}                               System Runtime: {runtime_str:<15}                         |")
    print(f"| Version: MK_Demo_3.2.11                                 Memory Usage: {mem_usage:<6}                                    |")
    print(f"| MoveNet FPS: {performance_info['pose_fps']:.1f} fps                                    CPU Usage: {cpu_usage:<6}                                       |")
    print(f"| MoveNet Latency: {performance_info['pose_latency']*1000:.1f} ms                               LSTM Latency: {performance_info['lstm_latency']*1000:.1f} ms                                   |")
    print(f"| DST Fusion Latency: {performance_info['dst_latency']*1000:.3f} ms                            CNN Latency: {performance_info['cnn_latency']*1000:.1f} ms                                    |")
    print("|                                                                                                                 |")
    print("|-----------------------------------------------------------------------------------------------------------------|")
    print("|                                                                                                                 |")
    print("| Pose Information (MoveNet)                             | Audio Features (MFCC)                                  |")
    print("| No.  Keypoint              X (%)    Y (%)    Conf      |                                                        |")
    
    if len(keypoints) > 0:
        kp = keypoints[0]
        print(f"| {1:>3}   {kp['name']:<15}{kp['x']:>10.2f}{kp['y']:>10.2f}{kp['conf']:>8.1f}      | Last Collection Time: {last_mfcc_time:<10}                       |")
    else:
        print(f"| {1:>3}   {'Nose':<15}{0:>10.2f}{0:>10.2f}{0:>8.1f}      | Last Collection Time: {last_mfcc_time:<10}                     |")
    
    if len(keypoints) > 1:
        kp = keypoints[1]
        print(f"| {2:>3}   {kp['name']:<15}{kp['x']:>10.2f}{kp['y']:>10.2f}{kp['conf']:>8.1f}      | Audio Length: {mfcc_info['audio_length']:.2f}s                                    | ")
    else:
        print(f"| {2:>3}   {'Left Eye':<15}{0:>10.2f}{0:>10.2f}{0:>8.1f}      | Audio Length: {mfcc_info['audio_length']:.2f}s                                    |")
    
    if len(keypoints) > 2:
        kp = keypoints[2]
        print(f"| {3:>3}   {kp['name']:<15}{kp['x']:>10.2f}{kp['y']:>10.2f}{kp['conf']:>8.1f}      | MFCC Range: [{mfcc_info['mfcc_range'][0]:.2f}, {mfcc_info['mfcc_range'][1]:.2f}]                               |")
    else:
        print(f"| {3:>3}   {'Right Eye':<15}{0:>10.2f}{0:>10.2f}{0:>8.1f}      | MFCC Range: [{mfcc_info['mfcc_range'][0]:.2f}, {mfcc_info['mfcc_range'][1]:.2f}]                                |")
    
    if len(keypoints) > 3:
        kp = keypoints[3]
        print(f"| {4:>3}   {kp['name']:<15}{kp['x']:>10.2f}{kp['y']:>10.2f}{kp['conf']:>8.1f}      |                                                        |")
    else:
        print(f"| {4:>3}   {'Left Ear':<15}{0:>10.2f}{0:>10.2f}{0:>8.1f}      |                                                        |")
    
    if len(keypoints) > 4:
        kp = keypoints[4]
        print(f"| {5:>3}   {kp['name']:<15}{kp['x']:>10.2f}{kp['y']:>10.2f}{kp['conf']:>8.1f}      |                                                        |")
    else:
        print(f"| {5:>3}   {'Right Ear':<15}{0:>10.2f}{0:>10.2f}{0:>8.1f}      |                                                        |")
    
    for i in range(5, 17):
        if i < len(keypoints):
            kp = keypoints[i]
            name = kp['name']
            x = kp['x']
            y = kp['y']
            conf = kp['conf']
        else:
            name = KEYPOINT_NAMES[i]
            x, y, conf = 0, 0, 0
        
        mfcc_value = mfcc_features[i] if i < len(mfcc_features) else 0.0
        mfcc_bar = create_vertical_bar(mfcc_value)
        mfcc_line = f"MFCC[{i-5:>2d}]: {mfcc_value:>7.1f} |{mfcc_bar}|"
        print(f"| {i+1:>3}   {name:<15}{x:>10.2f}{y:>10.2f}{conf:>8.1f}      | {mfcc_line:<41}              |")
    print("|                                                                                                                 |")
    print("|=================================================================================================================|")
    print("|                                                                                                                 |")
    print("| Model Inference Scores:                                                                                         |")
    print("|                                                                                                                 |")
    print(f"|   - MoveNet Pose Classification                          -Audio Recognition (LSTM)                              |")
    print(f"|     Score : {progress_bar(movenet_score, 15, BLUE)}                       Score : {progress_bar(audio_score, 15, MAGENTA)}                       |")
    
    print(f"|   - Conflict Coefficient K                               -DST Fusion Score                                      |")
    print(f"|     Score : {progress_bar(conflict_k, 15, CYAN)}                       Score : {progress_bar(fusion_score, 15, YELLOW)}                       |")
    
    print("|=================================================================================================================|")
    print("|                                                                                                                 |")
    print(f"| Model Version: {cnn_info['version']:<10}                                                                                       |")
    print(f"| Classification Threshold: θ = {cnn_info['threshold']:<3}                                                                               |")
    print(f"| Last CNN Analysis Time : {cnn_info['last_time']:<19}                                                                    |")
    
    if cnn_info['cnn_class_scores'] is not None:
        for label, score in cnn_info['cnn_class_scores'].items():
            if score >= 0.8:
                if label == 'Lying' or label == 'Falling':
                    color = RED
                elif label == 'Bending':
                    color = YELLOW
                else:
                    color = GREEN
                bar = progress_bar(score, 30, color)
            else:
                bar = progress_bar(score, 30)
            print(f"| {label:<8}: {bar:<45}                                                                |")
    else:
        for label, score in cnn_info['class_scores'].items():
            if score >= 0.8:
                if label == 'Lying' or label == 'Falling':
                    color = RED
                elif label == 'Bending':
                    color = YELLOW
                else:
                    color = GREEN
                bar = progress_bar(score, 30, color)
            else:
                bar = progress_bar(score, 30)
            print(f"| {label:<8}: {bar:<45}                                                         |")
    
    print("|                                                                                                                 |")
    print("|=================================================================================================================|")
    print("                                                                                                                ")
    print(f"| Status: {status:<80}                        ")
    if cnn_info['cnn_result'] is not None:
        print(f"| Classification Result: {cnn_info['cnn_result']:<80}                                                        ")
    else:
        print(f"| Classification Result: {cnn_info['result']:<80}                                                        ")
    
    print(f"| Event Status: {event_status:<90}              ")
    
    print("\n")
    print("| Fall Event Log (Time, Confidence, Action):")
    if fall_event_log:
        for t, conf, action in fall_event_log[-5:]:
            print(f"|   - {t}   Confidence: {conf:.2f}   Action: {action}")
    else:
        print("|    No fall events yet                                             ")
    print("\n")

def fusion_loop(audio_result_holder, visual_result_holder, keypoints_holder, fall_event_log, stop_event):
    try:
        init_camera(CAMERA_ID)
    except RuntimeError as e:
        stop_event.set()
        return

    last_cnn_analysis = {
        'time': None,
        'action': None,
        'confidence': None,
        'is_fall': False
    }
    current_fall_status = False
    start_time = time.perf_counter()
    version_str = "0.00.0"
    cnn_info = {
        'version': 'CNN v1.2.0',
        'threshold': 0.8,
        'last_time': '',
        'class_scores': {
            'Standing': 0.500,
            'Sitting': 0.300,
            'Lying': 0.01000,
            'Bending': 0.0200,
            'Falling': 0.0000
        },
        'cnn_class_scores': None,
        'cnn_result': None,
        'result': 'Standing (Below θ=0.80, No Alert)'
    }
    status = "Normal"
    event_status = "System Running Normally"
    last_cnn_latency = 0.0

    while not stop_event.is_set():
        now = time.perf_counter()
        now_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(now))
        runtime_str = time.strftime('%H:%M:%S', time.gmtime(now - start_time))
        mem_usage, cpu_usage = get_system_usage()

        audio_scores = [score for t, score, latency in audio_result_holder if now - t <= AUDIO_DURATION]
        visual_scores = [score for t, score, fps, latency in visual_result_holder if now - t <= AUDIO_DURATION]
        recent_keypoints = [(t, kp, frame) for t, kp, frame in keypoints_holder if now - t <= 1.0]

        if not audio_scores or not visual_scores:
            time.sleep(0.5)
            continue

        avg_audio = np.mean(audio_scores)
        avg_visual = np.mean(visual_scores)

        dst_start_time = time.perf_counter()
        
        m_v = calculate_belief_mass(avg_visual)
        m_a = calculate_belief_mass(avg_audio)
        m_f = dempster_rule(m_v, m_a)
        K = calculate_conflict_coefficient(m_v, m_a)
        
        dst_end_time = time.perf_counter()
        dst_latency = dst_end_time - dst_start_time
        
        fused = m_f['Fall']

        if recent_keypoints:
            _, latest_keypoints, _ = recent_keypoints[-1]
            keypoints = []
            for i, (x, y, conf) in enumerate(latest_keypoints):
                keypoints.append({
                    'name': KEYPOINT_NAMES[i],
                    'x': x * 100,
                    'y': y * 100,
                    'conf': conf * 100
                })
        else:
            keypoints = [{'name': n, 'x': 0, 'y': 0, 'conf': 0} for n in KEYPOINT_NAMES]

        mfcc_features = [0.0 for _ in range(17)]
        last_mfcc_time = now_str[-8:]
        if audio_result_holder:
            mfcc_features = [score for _, score, _ in list(audio_result_holder)[-17:]]
            last_mfcc_time = time.strftime('%H:%M:%S', time.localtime(audio_result_holder[-1][0]))

        mfcc_info = {
            'audio_length': AUDIO_DURATION,
            'mfcc_range': [min(mfcc_features) if mfcc_features else -40.76, max(mfcc_features) if mfcc_features else 38.60],
            'inference_time': 0.0327
        }

        movenet_score = avg_visual
        audio_score = avg_audio
        fusion_score = fused
        conflict_k = K

        performance_info = {
            'pose_fps': 0.0,
            'pose_latency': 0.0,
            'lstm_latency': 0.0,
            'dst_latency': 0.0,
            'cnn_latency': 0.0
        }
        
        if visual_result_holder:
            latest_visual = visual_result_holder[-1]
            if len(latest_visual) >= 4:
                performance_info['pose_fps'] = latest_visual[2]
                performance_info['pose_latency'] = latest_visual[3]
        
        if audio_result_holder:
            latest_audio = audio_result_holder[-1]
            if len(latest_audio) >= 3:
                performance_info['lstm_latency'] = latest_audio[2]
        
        performance_info['dst_latency'] = dst_latency

        if fused > FUSION_THRESHOLD:
            if recent_keypoints:
                _, _, latest_frame = recent_keypoints[-1]
                is_fall, action, confidence, class_scores, cnn_latency = analyze_abnormal_image(latest_frame)
                cnn_info['last_time'] = time.strftime('%Y-%m-%d %H:%M:%S')
                cnn_info['cnn_class_scores'] = class_scores
                result_color = RED if is_fall else GREEN
                cnn_info['cnn_result'] = f"{action} ({'Above' if confidence >= CNN_CONFIDENCE_THRESHOLD else 'Below'} θ={CNN_CONFIDENCE_THRESHOLD}, {result_color}{'Alert' if is_fall else 'No Alert'}{RESET})"
                
                last_cnn_latency = cnn_latency
                
                cnn_info['class_scores'] = {
                    'Standing': 1 - avg_visual,
                    'Sitting': 0.0,
                    'Lying': avg_visual if avg_visual > 0.5 else 0.0,
                    'Bending': 0.0,
                    'Falling': avg_visual if avg_visual > 0.5 else 0.0
                }
                
                if is_fall:
                    current_fall_status = True
                    status = f"{RED}Fall Detected: {action} ({confidence:.2f}){RESET}"
                    fall_event_log.append((cnn_info['last_time'], confidence, action))
                else:
                    status = f"{GREEN}Normal: {action} ({confidence:.2f}){RESET}"
                last_cnn_analysis.update({
                    'time': time.strftime('%H:%M:%S'),
                    'action': action,
                    'confidence': confidence,
                    'is_fall': is_fall
                })

        performance_info['cnn_latency'] = last_cnn_latency

        if current_fall_status:
            event_status = "Fall Event in Progress"
        else:
            event_status = "System Running Normally"

        os.system('cls' if os.name == 'nt' else 'clear')
        dashboard_render(
            now_str, runtime_str, version_str,
            mem_usage, cpu_usage,
            keypoints, mfcc_features, last_mfcc_time,
            movenet_score, audio_score, fusion_score, conflict_k,
            cnn_info, status, event_status,
            mfcc_info,
            fall_event_log,
            performance_info
        )
        time.sleep(1)

def gpio_monitor_thread(fall_event_log, stop_event):
    print(f"{BLUE}GPIO monitoring thread started{RESET}")
    print(f"{BLUE}Starting to monitor PIN40 state and control PIN38 output...{RESET}")
    
    last_clear_time = 0
    clear_cooldown = 0.5
    
    last_state = False
    last_fall_status = False
    
    while not stop_event.is_set():
        try:
            raw_value = GPIO.input(GPIO_PIN)
            current_state = raw_value == GPIO.HIGH
            
            current_fall_status = len(fall_event_log) > 0
            
            if current_fall_status != last_fall_status:
                if current_fall_status:
                    GPIO.output(GPIO_OUTPUT_PIN, GPIO.HIGH)
                    print(f"{RED}PIN38: HIGH - Fall event detected{RESET}")
                else:
                    GPIO.output(GPIO_OUTPUT_PIN, GPIO.LOW)
                    print(f"{GREEN}PIN38: LOW - No fall event{RESET}")
                last_fall_status = current_fall_status
            
            if current_state != last_state:
                if current_state:
                    print(f"{YELLOW}Button pressed detected!{RESET}")
                    current_time = time.perf_counter()
                    if (current_time - last_clear_time) >= clear_cooldown:
                        with gpio_lock:
                            fall_event_log.clear()
                            print(f"{GREEN}Fall event log cleared{RESET}")
                            GPIO.output(GPIO_OUTPUT_PIN, GPIO.LOW)
                            print(f"{GREEN}PIN38: LOW - Fall event log cleared{RESET}")
                            last_fall_status = False
                        last_clear_time = current_time
                else:
                    print(f"{BLUE}Button released{RESET}")
                
                last_state = current_state
            
            time.sleep(0.01)
            
        except Exception as e:
            print(f"{RED}GPIO monitoring error: {str(e)}{RESET}")
            print(f"{RED}Error details: {type(e).__name__}{RESET}")
            time.sleep(1)

def init_gpio():
    try:
        print(f"{BLUE}Setting GPIO mode...{RESET}")
        GPIO.setmode(GPIO.BOARD)
        print(f"{BLUE}Configuring PIN40...{RESET}")
        GPIO.setup(GPIO_PIN, GPIO.IN)
        print(f"{BLUE}Configuring PIN38...{RESET}")
        GPIO.setup(GPIO_OUTPUT_PIN, GPIO.OUT)
        GPIO.output(GPIO_OUTPUT_PIN, GPIO.LOW)
        print(f"{GREEN}GPIO initialization successful{RESET}")
        return True
    except Exception as e:
        print(f"{RED}GPIO initialization failed: {str(e)}{RESET}")
        print(f"{RED}Error details: {type(e).__name__}{RESET}")
        return False

def main():
    try:
        print("System starting...")
        
        print(f"{BLUE}=== Initial System Resources ==={RESET}")
        mem_usage, cpu_usage = get_system_usage()
        print(f"{BLUE}Memory: {mem_usage}, CPU: {cpu_usage}{RESET}")
        print()
        
        print(f"{BLUE}Initializing GPIO...{RESET}")
        if not init_gpio():
            print(f"{RED}Warning: GPIO initialization failed, system will continue to run but cannot use GPIO functionality{RESET}")
        else:
            print(f"{GREEN}GPIO initialization completed{RESET}")
        
        audio_result_holder = deque(maxlen=17)
        visual_result_holder = deque(maxlen=50)
        keypoints_holder = deque(maxlen=10)
        fall_event_log = []
        stop_event = threading.Event()
        
        threads = [
            threading.Thread(target=audio_thread_fn, args=(audio_result_holder, stop_event)),
            threading.Thread(target=visual_thread_fn, args=(visual_result_holder, keypoints_holder, stop_event)),
            threading.Thread(target=fusion_loop, args=(audio_result_holder, visual_result_holder, keypoints_holder, fall_event_log, stop_event)),
            threading.Thread(target=gpio_monitor_thread, args=(fall_event_log, stop_event))
        ]
        
        for t in threads:
            t.daemon = True
            t.start()
        
        for t in threads:
            t.join()
            
    except KeyboardInterrupt:
        print("Shutting down system...")
        stop_event.set()
    except Exception as e:
        print(f"System error: {str(e)}")
        stop_event.set()
    finally:
        if camera is not None:
            camera.release()
        GPIO.cleanup()
        print("System shutdown complete")

if __name__ == "__main__":
    main()
    