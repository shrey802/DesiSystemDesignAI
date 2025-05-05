# DesiSystemDesignAI

A system design RAG model that takes Hindi input from finetuned Whisper ASR model and translates via Groq API then performs Pinecone search query and formats answer using Llama model and produces wav file using Speech T5 finetuned on more English


ASR - WHISPER TINY 
HINDI TO ENGLISH - GROQ 
Q&A VECTORS - PINECONE DB 
LLMS & RETRIEVAL - LLAMA 
TTS - Speech T5


(ASR → Query → Vector DB → LLM → TTS)


COLAB CODE TO AVOID RESTART SESSION
<!-- function ConnectButton(){
    console.log("Connect pushed");
    document.querySelector("#top-toolbar > colab-connect-button").shadowRoot.querySelector("#connect").click()
}
setInterval(ConnectButton, 60000); -->



