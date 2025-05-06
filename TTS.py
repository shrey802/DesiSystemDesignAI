






from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
import soundfile as sf
import torch
import torchaudio
import pygame

MODEL_ID = "hijklmno/speecht5_finetuned_sysdesign"

# Initialize processor, model, and vocoder
processor = SpeechT5Processor.from_pretrained(MODEL_ID)
model = SpeechT5ForTextToSpeech.from_pretrained(MODEL_ID).to("cuda" if torch.cuda.is_available() else "cpu")
vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan").to(model.device)

speaker_embeddings = torch.randn(1, 512).to(model.device)

def synthesize(output_file="answer_output.wav"):
    try:
        with open('final_output.txt', 'r', encoding='utf-8') as final:
            text = final.read().strip()
            print(f"📄 Text to synthesize: {text}")
    except Exception as e:
        print(f"⚠️ Error reading final_output.txt: {e}")
        return

    # Tokenize input
    inputs = processor(text=text, return_tensors="pt", padding=True, truncation=True).to(model.device)

    # Generate speech
    try:
        speech = model.generate_speech(inputs["input_ids"], speaker_embeddings, vocoder=vocoder)
        if speech.ndimension() == 2:
            speech = speech.squeeze(0)
    except Exception as e:
        print(f"⚠️ Error during speech generation: {e}")
        return

    # Save to file
    try:
        sf.write(output_file, speech.cpu().numpy(), samplerate=16000)
        print(f"✅ Audio saved as {output_file}")
    except Exception as e:
        print(f"⚠️ Error saving audio file: {e}")
        return

    # Play audio with pygame
    try:
        pygame.mixer.init()
        pygame.mixer.music.load(output_file)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(20)
    except Exception as e:
        print(f"⚠️ Error during audio playback: {e}")



synthesize("answer_output.wav")
