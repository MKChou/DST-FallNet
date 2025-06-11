import torch
import torchaudio.transforms as T
import sounddevice as sd
import onnxruntime as ort
import time
import threading
from collections import deque
import numpy as np

SAMPLE_RATE = 16000
AUDIO_DURATION = 3
CHANNELS = 1

audio_lock = threading.Lock()

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
        session = ort.InferenceSession("FallFusion-Audio.onnx", providers=["CPUExecutionProvider"])
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name
    except Exception as e:
        print(f"Audio model loading failed: {str(e)}")
        return

    while not stop_event.is_set():
        try:
            waveform = record_audio()
            mfcc = extract_mfcc(waveform)
            
            inference_start_time = time.perf_counter()
            output = session.run([output_name], {input_name: mfcc})[0][0]
            inference_end_time = time.perf_counter()
            
            inference_latency = inference_end_time - inference_start_time
            
            score = output[0] if len(output) == 1 else np.max(output)
            with audio_lock:
                audio_result_holder.append((time.perf_counter(), score, inference_latency))
        except Exception as e:
            print(f"Audio processing failed: {str(e)}")
        time.sleep(0.1) 