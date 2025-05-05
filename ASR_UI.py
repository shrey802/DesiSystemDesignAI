import sounddevice as sd
from scipy.io.wavfile import write
import torchaudio
import torch
from transformers import pipeline
 
 # Step 1: Load your Whisper fine-tuned Hindi model
pipe = pipeline("automatic-speech-recognition", model="yungcodedev/whisper-small-hi", device=0)
 
 # Step 2: Record your voice and save as a .wav file
def record_and_save(filename="recorded.wav", duration=5, sample_rate=16000):
     
     audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float32')
     sd.wait()
     
     # Save the audio file
     write(filename, sample_rate, audio)
 
 # Step 3: Transcribe the saved .wav file
def transcribe_audio(file_path):
     waveform, sample_rate = torchaudio.load(file_path)
 
     # Convert to mono if needed
     if waveform.shape[0] > 1:
         waveform = waveform.mean(dim=0)
 
     # Resample if not 16000 Hz
     if sample_rate != 16000:
         waveform = torchaudio.functional.resample(waveform, sample_rate, 16000)
         sample_rate = 16000
 
     result = pipe({
         "array": waveform.squeeze().numpy(),
         "sampling_rate": sample_rate
     })
     transcription = result["text"]
     with open("output.txt", "w", encoding="utf-8") as f:
         f.write(transcription)
     
 
record_and_save("my_voice.wav", duration=10)
transcribe_audio("my_voice.wav")