import pyaudio
import numpy as np
import signal
import sys

# Parameters for audio capture
FORMAT = pyaudio.paInt16  # Audio format (16-bit)
CHANNELS = 1              # Mono audio
RATE = 44100              # Sample rate (44.1kHz)
CHUNK = 2048              # Number of frames per buffer (increased for stability)
SAMPLE_WIDTH = 2          # Sample width in bytes (2 bytes for 16-bit)
TARGET_LEVEL = 1000       # Target RMS level for AGC

# Initialize PyAudio
p = pyaudio.PyAudio()

# Open the microphone stream for input and output
stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                output=True,
                frames_per_buffer=CHUNK)

print("Recording... Press Ctrl+C to stop.")

# Function to calculate RMS (Root Mean Square) value of audio data
def calculate_rms(data):
    return np.sqrt(np.mean(np.square(data)))

# Function to apply gain to the audio data
def apply_gain(data, gain_factor):
    return np.clip(data * gain_factor, -32768, 32767).astype(np.int16)

# Function to handle graceful exit on Ctrl+C
def signal_handler(sig, frame):
    print('Exiting...')
    stream.stop_stream()
    stream.close()
    p.terminate()
    sys.exit(0)

# Register signal handler for Ctrl+C
signal.signal(signal.SIGINT, signal_handler)

# Real-time audio processing loop
while True:
    try:
        # Capture a chunk of audio data
        audio_data = np.frombuffer(stream.read(CHUNK), dtype=np.int16)
        
        # Calculate the RMS of the audio data to determine loudness
        rms = calculate_rms(audio_data)
        
        # Check if RMS is zero (avoid division by zero)
        if rms == 0:
            gain_factor = 1.0  # If RMS is 0, don't apply any gain
        else:
            # Calculate the gain factor (if RMS is too low or too high)
            gain_factor = TARGET_LEVEL / rms
        
        # Prevent gain factor from becoming NaN or inf
        if np.isnan(gain_factor) or np.isinf(gain_factor):
            gain_factor = 1.0  # Default gain if NaN or infinity

        # Apply the gain factor to the audio data
        adjusted_audio = apply_gain(audio_data, gain_factor)
        
        # Send the adjusted audio back to the output (speakers)
        stream.write(adjusted_audio.tobytes())
        
    except IOError as e:
        # Handle overflow error (buffer overflow)
        if e.errno == -9981:
            print("Buffer overflow detected! Retrying...")
            continue  # Skip this iteration and try again
        else:
            raise e  # Reraise unexpected errors
