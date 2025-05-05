# from pinecone import Pinecone
# from langchain_community.embeddings import HuggingFaceEmbeddings
# import os
# import requests
# from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
# from huggingface_hub import login

# login("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") 

# os.environ['PINECONE_API_KEY'] = "pcsk_JbTgY_8nL8BcQhif14xgerADD1WhLgbJLbZnPzZ5KzsNkfnWYPH6nkaZMFRajhjGs38bg"
# pc = Pinecone(api_key=os.environ['PINECONE_API_KEY'])
# GROQ_API_KEY = "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"
# index = pc.Index("desi-design-index")
# embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


# def query_pinecone(user_query, top_k=5, namespace=None):
#     query_vector = embedding_model.embed_query(user_query)
#     results = index.query(
#         vector=query_vector,
#         top_k=top_k,
#         include_metadata=True,
#         namespace=namespace
#     )

#     for match in results["matches"]:
#         print(match['metadata']['text'])  
#     return results["matches"]

# # --- Groq-based Answer Generator ---
# def generate_answer_with_groq(context_chunks, user_query):
#     context = "\n\n".join([match['metadata']['text'] for match in context_chunks])
#     prompt = f"""You are an assistant with deep knowledge of system design.
# Use the following context to answer the user's question as clearly and precisely as possible from the context provided and in 200 characters.

# ### Context:
# {context}

# ### Question:
# {user_query}

# ### Answer:
# """

#     headers = {
#         "Authorization": f"Bearer {GROQ_API_KEY}",
#         "Content-Type": "application/json"
#     }

#     data = {
#         "model": "llama-3.3-70b-versatile",  
#         "messages": [
#             {"role": "system", "content": "You are a helpful assistant."},
#             {"role": "user", "content": prompt}
#         ],
#         "temperature": 0.7,
#         "max_tokens": 512
#     }

#     # Send the request to Groq
#     response = requests.post(
#         "https://api.groq.com/openai/v1/chat/completions",
#         headers=headers,
#         json=data
#     )

#     # Return the generated answer
#     return response.json()['choices'][0]['message']['content'].strip()
# query_query = ''
# with open('translated_output.txt', 'r') as file:
#     query_query = ''.join(file.readlines())
# user_query = query_query
# retrieved_chunks = query_pinecone(user_query, top_k=5, namespace="systemdesign1")  
# answer = generate_answer_with_groq(retrieved_chunks, user_query)  

# with open('final_output.txt','w') as finalfile:
#     finalfile.write(answer.strip())


















# query_pinecone.py
from pinecone import Pinecone
from langchain_community.embeddings import HuggingFaceEmbeddings
import requests
import os

# Initialize Pinecone and embeddings
os.environ['PINECONE_API_KEY'] = "pcsk_JbTgY_8nL8BcQhif14xgerADD1WhLgbJLbZnPzZ5KzsNkfnWYPH6nkaZMFRajhjGs38bg"
pc = Pinecone(api_key=os.environ['PINECONE_API_KEY'])
GROQ_API_KEY = "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"
index = pc.Index("desi-design-index")
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


def query_pinecone(user_query, top_k=5, namespace=None):
    query_vector = embedding_model.embed_query(user_query)
    results = index.query(
        vector=query_vector,
        top_k=top_k,
        include_metadata=True,
        namespace=namespace
    )

    for match in results["matches"]:
        print(match['metadata']['text'])  
    return results["matches"]

def generate_answer_with_groq(context_chunks, user_query):
    context = "\n\n".join([match['metadata']['text'] for match in context_chunks])
    prompt = f"""You are an assistant with deep knowledge of system design.
Use the following context to answer the user's question as clearly and precisely as possible from the context provided in 5 lines.

### Context:
{context}

### Question:
{user_query}

### Answer:
"""

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    data = {
        "model": "llama-3.3-70b-versatile",  
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7,
        "max_tokens": 512
    }

    # Send the request to Groq
    response = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers=headers,
        json=data
    )

    # Return the generated answer
    return response.json()['choices'][0]['message']['content'].strip()

def process_query(file_path, top_k=5, namespace="systemdesign1"):
    # Read the query text from a file
    with open(file_path, 'r') as file:
        query_query = ''.join(file.readlines())

    # Get retrieved chunks from Pinecone
    retrieved_chunks = query_pinecone(query_query, top_k, namespace)

    # Generate an answer using Groq
    answer = generate_answer_with_groq(retrieved_chunks, query_query)
    
    # Write the answer to the final output file
    with open('final_output.txt', 'w') as finalfile:
        finalfile.write(answer.strip())

    print("Answer generated and saved in final_output.txt")

process_query('translated_output.txt')