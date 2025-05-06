from ASR_UI import record_and_save, transcribe_audio  
from translate import translate  
from retrieval import process_query
from TTS import synthesize
import pygame
import time

def main():
    print("🎙️ Welcome to the CLI Voice Assistant. Press 'q' to quit.\n")

    while True:
        user_input = input("🔁 Press Enter to speak (or 'q' to quit): ").strip().lower()
        if user_input == 'q':
            print("👋 Exiting...")
            break

        # Step 1: ASR (Record and transcribe)
        print("🎤 Recording...")
        record_and_save("my_voice.wav", duration=10)

        print("📝 Transcribing speech...")
        transcription = transcribe_audio("my_voice.wav")
        print(f"📝 Transcribed (Hindi): {transcription}")

        # Save transcription to output.txt (important step)
        with open("output.txt", "w", encoding="utf-8") as f:
            f.write(transcription)

        # Step 2: Translate the transcription from output.txt to English
        print("🌐 Translating to English...")

        try:
            # Call the translate function
            translate()
            print("🌐 Translation complete! Translated content saved in 'translated_output.txt'\n")
        except Exception as e:
            print(f"⚠️ Error during translation: {e}")

        try:
            process_query('translated_output.txt')
        except Exception as e:
            print(f"⚠️ Error during translation: {e}")

        try:
            synthesize("answer_output.wav")
        except Exception as e:
            print(f"Error during {e}")


if __name__ == "__main__":
    print("🔄 Starting the CLI Voice Assistant...")
    main()
