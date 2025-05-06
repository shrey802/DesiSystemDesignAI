 
# Desi System Design AI 🧠 

A fully voice-driven System Design RAG (Retrieval-Augmented Generation) pipeline in Hindi — powered by Whisper, Groq, Pinecone, LLaMA, and SpeechT5 built as a side project.


## Technology Used :-

ASR	Whisper Tiny (finetuned)	Finetuned for 2000 epochs (https://discuss.huggingface.co/t/whisper-model-fine-tuning/26045)

Translation	Groq API	For fast Hindi-to-English conversion

Vector DB	Pinecone	6500+ curated Q&A vectors

LLM	LLaMA 3.3 70B Versatile	Generates context-aware responses

TTS	SpeechT5 (finetuned)	Trained for 1500 epochs on English speech



## 🔁 Google Colab Auto-Reconnect

To avoid session timeout during long model training in Google Colab, use the following JavaScript snippet in your browser's developer console:

```javascript
// 🔁 Auto-reconnect for Google Colab

function ConnectButton() {
    console.log("Connect pushed");
    document.querySelector("#top-toolbar > colab-connect-button")
        .shadowRoot.querySelector("#connect").click();
}
setInterval(ConnectButton, 60000);





