import os
from utils import (
    RED, GREEN, YELLOW, BLUE, MAGENTA, CYAN, RESET,
    progress_bar, create_vertical_bar
)

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