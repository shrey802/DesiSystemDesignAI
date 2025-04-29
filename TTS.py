from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
import soundfile as sf
import torch 
import torchaudio

MODEL_ID = "hijklmno/speecht5_finetuned_sysdesign"

processor = SpeechT5Processor.from_pretrained(MODEL_ID)
model = SpeechT5ForTextToSpeech.from_pretrained(MODEL_ID).to("cuda" if torch.cuda.is_available() else "cpu")
vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan").to(model.device)

speaker_embeddings = torch.randn(1, 512).to(model.device) 

def synthesize(text, output_file="output.wav"):
    inputs = processor(text=text, return_tensors="pt").to(model.device)
    speech = model.generate_speech(inputs["input_ids"], speaker_embeddings, vocoder=vocoder)
    sf.write(output_file, speech.cpu().numpy(), samplerate=16000)
    print(f"✅ Audio saved as {output_file}")

# Example long paragraph
paragraph = """
The client sends a request to the LoadBalancer, which then routes the request to one of the services – A, B, or C – depending on availability and load.
Each service is stateless, meaning it can handle requests independently without relying on previous data.
This design ensures high availability, fault tolerance, and easy scaling.
"""

synthesize(paragraph, "answer_output.wav")