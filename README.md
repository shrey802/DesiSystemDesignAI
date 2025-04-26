# DesiSystemDesignAI

A system design AI agent to answer your system design related queries and give recommendation also provide an architecture diagram & all this using voice in Hindi. 


ASR - WHISPER TINY FINE TUNED (DONE)
HINDI TO ENGLISH GROQ TRANSLATION (DONE)
Q&A VECTORS - PINECONE DB (DONE)
LLMS & RETRIEVAL - LLAMA / MISTRAL / GROQ
TTS - [TO BE DECIDED](https://colab.research.google.com/drive/1i7I5pzBcU3WDFarDnzweIj4-sVVoIUFJ)
DIAGRAM GENERATION - TO BE DECIDED


(ASR → Query → Vector DB → LLM → Diagram Generation → TTS)


COLAB CODE TO AVOID RESTART SESSION
<!-- function ConnectButton(){
    console.log("Connect pushed");
    document.querySelector("#top-toolbar > colab-connect-button").shadowRoot.querySelector("#connect").click()
}
setInterval(ConnectButton, 60000); -->


<!-- from transformers import pipeline
import torchaudio

# Load your Whisper Hindi fine-tuned model
pipe = pipeline("automatic-speech-recognition", model="yungcodedev/whisper-small-hi")

# Load the audio file
file_path = "download.wav"
waveform, sample_rate = torchaudio.load(file_path)

# Stereo to mono
if waveform.shape[0] > 1:
    waveform = waveform.mean(dim=0)

# Resample if needed
if sample_rate != 16000:
    waveform = torchaudio.functional.resample(waveform, sample_rate, 16000)
    sample_rate = 16000

# Run ASR with correct input format
result = pipe({
    "array": waveform.squeeze().numpy(),
    "sampling_rate": sample_rate
})
print("🗣️ Transcription:", result["text"]) -->



<!-- User Input (Speech in Hindi):

The user speaks in Hindi to query the system.

Example: "लोड बैलेंसर क्या है?"

ASR (Whisper):

Convert the speech to Hindi text: "लोड बैलेंसर क्या है?"

Translation (Hindi to English):

Translate the Hindi query to English: "What is a load balancer?"

Pinecone Query:

Embed the translated query and query Pinecone for relevant documents or answers.

Answer Generation (LLM):

Generate a detailed response about load balancers using Gemini or Groq.

Translation (English to Hindi):

Translate the generated answer back to Hindi: "लोड बैलेंसर एक उपकरण है जो ट्रैफिक को वितरित करके उच्च उपलब्धता सुनिश्चित करता है।"

TTS (Text-to-Speech):

Convert the final Hindi text to speech and deliver the answer audibly.

Mermaid Diagram:

Optionally, generate a Mermaid diagram if the query requires a system design explanation with a visual representation. -->