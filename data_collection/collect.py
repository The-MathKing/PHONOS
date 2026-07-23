import asyncio
import websockets
import json
import sounddevice as sd
import numpy as np
import parselmouth
from parselmouth.praat import call
import time
import os
import threading

# Configuration
WS_URI = "ws://localhost:8080"
SAMPLE_RATE = 44100
DURATION = 10  # Seconds per recording block
DATA_DIR = "dataset"

if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# Global Buffers
bio_data_buffer = []
is_recording = False

async def bio_stream_client():
    """ Connects to the Node.js fast-parser WebSocket and logs biological telemetry """
    global is_recording, bio_data_buffer
    try:
        async with websockets.connect(WS_URI) as websocket:
            print(f"[+] Connected to biological telemetry stream at {WS_URI}")
            async for message in websocket:
                if is_recording:
                    # Message format: {"phoneticVector": [A0, A1, A2, f3, f4, f5], "affect": {"arousal": a, "valence": v}}
                    # Or raw array depending on phase. We expect raw DSP output for training.
                    try:
                        data = json.loads(message)
                        # We need timestamped logs to align with audio
                        timestamp = time.time()
                        bio_data_buffer.append((timestamp, data))
                    except json.JSONDecodeError:
                        pass
    except Exception as e:
        print(f"[-] WebSocket connection error: {e}")

def extract_acoustic_features(audio_data, sr):
    """ Uses Praat-Parselmouth to extract exact clinical Formants and f0 """
    # Convert numpy array to parselmouth Sound object
    snd = parselmouth.Sound(audio_data, sampling_frequency=sr)
    
    # Extract f0 (Pitch)
    pitch = snd.to_pitch()
    pitch_values = pitch.selected_array['frequency']
    
    # Extract Formants (F1, F2)
    formants = snd.to_formant_burg(max_number_of_formants=5, maximum_formant=5500.0)
    
    num_frames = pitch.get_number_of_frames()
    
    f0_contour = []
    f1_contour = []
    f2_contour = []
    timestamps = []
    
    for i in range(1, num_frames + 1):
        t = pitch.get_time_from_frame_number(i)
        f0 = pitch.get_value_at_time(t)
        f1 = formants.get_value_at_time(1, t)
        f2 = formants.get_value_at_time(2, t)
        
        # Filter unvoiced/NaN
        f0 = f0 if not np.isnan(f0) else 0.0
        f1 = f1 if not np.isnan(f1) else 0.0
        f2 = f2 if not np.isnan(f2) else 0.0
        
        timestamps.append(t)
        f0_contour.append(f0)
        f1_contour.append(f1)
        f2_contour.append(f2)
        
    return np.array(timestamps), np.array(f0_contour), np.array(f1_contour), np.array(f2_contour)

def process_and_align_session(bio_data_log, audio_path, sample_rate=2000):
    """
    Synchronizes high-speed 4-channel bio telemetry matrices with 
    clinical acoustic targets extracted via Praat-Parselmouth.
    """
    # 1. Load sound file into Parselmouth
    sound = parselmouth.Sound(audio_path)
    
    # 2. Extract continuous clinical features via Praat cross-correlation
    pitch = sound.to_pitch()
    formants = sound.to_formant_burg(time_step=0.005) # 5ms windows matching ML steps
    
    aligned_dataset = []
    
    # 3. Step through the timeline and index time-locked values
    for packet in bio_data_log:
        t = packet['timestamp']
        
        # Pull bio values (Normalized by your 5-second firmware calibration)
        inputs = [packet['ch1_emg'], packet['ch2_emg'], packet['pzt'], packet['eda']]
        
        # Synchronize corresponding acoustic features
        f0 = pitch.get_value_at_time(t)
        f1 = formants.get_value_at_time(1, t)
        f2 = formants.get_value_at_time(2, t)
        
        # Handle unvoiced segments (Praat returns NaN if no pitch detected)
        f0_clean = 0.0 if np.isnan(f0) else f0
        f1_clean = 500.0 if np.isnan(f1) else f1 # Default fallback baseline
        f2_clean = 1500.0 if np.isnan(f2) else f2
        
        # Construct row: [Inputs (4)] -> [Phonetic Targets (2 Formants...)] -> [Affective Targets (f0...)]
        aligned_dataset.append(inputs + [f1_clean, f2_clean, f0_clean])
        
    return np.array(aligned_dataset)

def record_audio_and_bio():
    """ Handles synchronous audio recording and triggers bio logging """
    global is_recording, bio_data_buffer
    
    print(f"\n[!] Preparing to record for {DURATION} seconds...")
    time.sleep(2)
    
    bio_data_buffer = []
    start_time = time.time()
    
    print("[*] RECORDING START - Speak your target phonetics!")
    is_recording = True
    
    # Record audio synchronously
    audio_data = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='float32')
    sd.wait()
    
    is_recording = False
    print("[*] RECORDING STOPPED")
    
    audio_data = audio_data.flatten()
    
    # Process Acoustic Targets via Praat
    print("[+] Extracting clinical phonetic metrics (f0, F1, F2)...")
    a_time, f0, f1, f2 = extract_acoustic_features(audio_data, SAMPLE_RATE)
    
    # Shift acoustic timestamps to global unix time for alignment
    a_time += start_time
    
    # Save raw biological arrays and target features
    timestamp_str = str(int(time.time()))
    bio_file = os.path.join(DATA_DIR, f"bio_{timestamp_str}.npy")
    acoustic_file = os.path.join(DATA_DIR, f"acoustic_{timestamp_str}.npy")
    
    # We will save the biological data as a structured array or dict
    np.save(bio_file, np.array(bio_data_buffer, dtype=object))
    
    # Acoustic targets: [time, f0, F1, F2]
    acoustic_targets = np.column_stack((a_time, f0, f1, f2))
    np.save(acoustic_file, acoustic_targets)
    
    print(f"[+] Dataset saved: {bio_file} and {acoustic_file}")

async def main():
    # Run the websocket client in the background
    client_task = asyncio.create_task(bio_stream_client())
    
    # Give it a second to connect
    await asyncio.sleep(1)
    
    # Run a recording block in a separate thread so it doesn't block the async loop
    recording_thread = threading.Thread(target=record_audio_and_bio)
    recording_thread.start()
    
    while recording_thread.is_alive():
        await asyncio.sleep(0.1)
        
    client_task.cancel()

if __name__ == "__main__":
    print("=== CADENCE Clinical Data Alignment Interface ===")
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n[!] Exiting...")
