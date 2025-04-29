import sounddevice as sd
from scipy.io.wavfile import write
import torchaudio
import torch
import numpy as np
from transformers import pipeline
import os

# Step 1: Load your Whisper fine-tuned Hindi model
# Only load it when needed to avoid CUDA memory issues
def load_model():
    print("Loading Whisper model...")
    return pipeline("automatic-speech-recognition", model="yungcodedev/whisper-small-hi", device=0)

# Step 2: Record your voice and save as a .wav file
def record_and_save(filename="recorded.wav", duration=5, sample_rate=16000):
    print(f"🎙️ Recording for {duration} seconds... Speak now!")
    
    # Record audio
    audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float32')
    sd.wait()
    print("✅ Recording complete.")
    
    # Convert to int16 for better compatibility
    audio_int16 = (audio * 32767).astype(np.int16)
    
    # Save the audio file - this line was commented out in your original code
    write(filename, sample_rate, audio_int16)
    print(f"✅ Audio saved to {filename}")
    
    # Verify the file was created
    if os.path.exists(filename):
        print(f"✅ File size: {os.path.getsize(filename)} bytes")
    else:
        print("❌ File was not created successfully!")

# Step 3: Transcribe the saved .wav file
def transcribe_audio(file_path):
    print(f"Transcribing audio from {file_path}...")
    
    # Check if file exists
    if not os.path.exists(file_path):
        print(f"❌ File {file_path} does not exist!")
        return
    
    try:
        # Load the audio using torchaudio
        waveform, sample_rate = torchaudio.load(file_path)
        
        # Convert to mono if needed
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        
        # Resample if not 16000 Hz
        if sample_rate != 16000:
            waveform = torchaudio.functional.resample(waveform, sample_rate, 16000)
            sample_rate = 16000
        
        # Load model only when needed
        pipe = load_model()
        
        # Prepare input for Whisper
        input_features = {
            "array": waveform.squeeze().numpy(),
            "sampling_rate": sample_rate
        }
        
        # Run transcription
        result = pipe(input_features)
        transcription = result["text"]
        
        print(f"Transcription: {transcription}")
        
        # Save transcription to file
        with open("output.txt", "w", encoding="utf-8") as f:
            f.write(transcription)
        
        print("✅ Transcription saved to output.txt")
        return transcription
    
    except Exception as e:
        print(f"❌ Error during transcription: {str(e)}")
        print("Let's try an alternative approach...")
        
        try:
            # Alternative approach using scipy to read the file
            from scipy.io import wavfile
            sample_rate, audio_data = wavfile.read(file_path)
            
            # Convert to float32 and normalize
            if audio_data.dtype == np.int16:
                audio_data = audio_data.astype(np.float32) / 32767.0
            
            # Ensure mono
            if len(audio_data.shape) > 1 and audio_data.shape[1] > 1:
                audio_data = np.mean(audio_data, axis=1)
            
            # Load model if not loaded
            if 'pipe' not in locals():
                pipe = load_model()
            
            # Run transcription
            result = pipe({"array": audio_data, "sampling_rate": sample_rate})
            transcription = result["text"]
            
            print(f"Transcription (alt method): {transcription}")
            
            # Save transcription to file
            with open("output.txt", "w", encoding="utf-8") as f:
                f.write(transcription)
            
            print("✅ Transcription saved to output.txt")
            return transcription
            
        except Exception as e2:
            print(f"❌ Alternative approach also failed: {str(e2)}")
            return None

# Main execution
if __name__ == "__main__":
    # Define file path
    audio_file = "my_voice.wav"
    
    # Record audio
    record_and_save(audio_file, duration=10)
    
    # Transcribe audio
    if os.path.exists(audio_file):
        transcription = transcribe_audio(audio_file)
    else:
        print("❌ Recording failed. File not found.")