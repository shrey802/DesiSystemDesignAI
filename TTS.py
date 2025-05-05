






from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
import soundfile as sf
import torch
import torchaudio

MODEL_ID = "hijklmno/speecht5_finetuned_sysdesign"

# Initialize processor, model, and vocoder
processor = SpeechT5Processor.from_pretrained(MODEL_ID)
model = SpeechT5ForTextToSpeech.from_pretrained(MODEL_ID).to("cuda" if torch.cuda.is_available() else "cpu")
vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan").to(model.device)

speaker_embeddings = torch.randn(1, 512).to(model.device)

# Function to synthesize speech from text
def synthesize(text, output_file="output.wav"):
    # Process input text
    inputs = processor(text=text, return_tensors="pt", padding=True, truncation=True).to(model.device)
    
    # Generate speech
    try:
        speech = model.generate_speech(inputs["input_ids"], speaker_embeddings, vocoder=vocoder)
    except Exception as e:
        print(f"Error during speech generation: {e}")
        return
    
    if speech.ndimension() == 2:  
        speech = speech.squeeze(0)

    try:
        sf.write(output_file, speech.cpu().numpy(), samplerate=16000)
        print(f"✅ Audio saved as {output_file}")
    except Exception as e:
        print(f"Error saving audio file: {e}")


paragraph = ""
with open('final_output.txt', 'r') as final:
    paragraph = final.read().strip()

synthesize(paragraph, "answer_output.wav")
