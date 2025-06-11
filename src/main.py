import threading
import time
from collections import deque
import RPi.GPIO as GPIO

from utils import get_system_usage, BLUE, RED, GREEN, RESET
from audio_module import audio_thread_fn, AUDIO_DURATION
from visual_module import visual_thread_fn, init_camera, CAMERA_ID, KEYPOINT_NAMES
from fusion import calculate_belief_mass, dempster_rule, calculate_conflict_coefficient
from cnn_analysis import analyze_abnormal_image, CNN_CONFIDENCE_THRESHOLD
from dashboard import dashboard_render
from gpio_control import init_gpio, gpio_monitor_thread

FUSION_THRESHOLD = 0.5

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

        avg_audio = sum(audio_scores) / len(audio_scores)
        avg_visual = sum(visual_scores) / len(visual_scores)

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
        if 'camera' in globals() and camera is not None:
            camera.release()
        GPIO.cleanup()
        print("System shutdown complete")

if __name__ == "__main__":
    main() 