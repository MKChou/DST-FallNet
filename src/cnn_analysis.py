import cv2
import numpy as np
import onnxruntime as ort
import time
from utils import clear_gpu_memory

CNN_MODEL_PATH = "FallFusion-CNN.onnx"
CNN_CONFIDENCE_THRESHOLD = 0.8

CNN_LABELS = {
    0: "standing",
    1: "sitting",
    2: "lying",
    3: "bending",
    4: "crawling"
}

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