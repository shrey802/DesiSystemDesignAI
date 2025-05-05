# Chunk size is the maximum number of characters that a chunk can contain. 
# Chunk overlap is the number of characters that should overlap between two adjacent chunks.

# recursivecharactertextsplitter is a function that splits into small chunk size automatically and is useful only if we need fine grained view like search engine optimization and keywords
# giving chunk size and chunk overlap manually is useful for overall view like Q&A

# PINECONE_API_KEY = "pcsk_JbTgY_8nL8BcQhif14xgerADD1WhLgbJLbZnPzZ5KzsNkfnWYPH6nkaZMFRajhjGs38bg"

# from pinecone import (
#     Pinecone,
#     ServerlessSpec
# )
# from langchain_community.document_loaders import PyPDFLoader
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain_community.embeddings import HuggingFaceEmbeddings
# import os

# # Initialize Pinecone client
# os.environ['PINECONE_API_KEY'] = "pcsk_JbTgY_8nL8BcQhif14xgerADD1WhLgbJLbZnPzZ5KzsNkfnWYPH6nkaZMFRajhjGs38bg"
# pc = Pinecone(api_key=os.environ['PINECONE_API_KEY'])

# index_name = "desi-design-index"

# # Create index if it doesn't exist
# if index_name not in [index.name for index in pc.list_indexes()]:
#     pc.create_index(
#         name=index_name,
#         dimension=384,  # for sentence-transformers/all-MiniLM-L6-v2
#         metric="cosine",
#         spec=ServerlessSpec(
#             cloud="aws",
#             region="us-east-1"
#         )
#     )

# # Connect to existing index
# index = pc.Index(index_name)

# # Initialize HuggingFace embedding model
# embeddings = HuggingFaceEmbeddings(
#     model_name="sentence-transformers/all-MiniLM-L6-v2",
#     model_kwargs={"device": "cpu"}  # Use "cuda" if you have GPU
# )

# # Text splitter
# text_splitter = RecursiveCharacterTextSplitter(
#     chunk_size=1000,
#     chunk_overlap=200
# )

# # List of PDFs
# pdf_paths = [
#     "Data/Designing Data-Intensive Applications The Big Ideas Behind Reliable, Scalable, and Maintainable Systems.pdf",
#     "Data/grok_system_design_interview.pdf",
#     "Data/System Design Interview An Insider’s Guide.pdf"
# ]

# # Process each PDF
# for pdf_path in pdf_paths:
#     print(f"Processing {pdf_path}...")
    
#     loader = PyPDFLoader(pdf_path)
#     documents = loader.load()
#     docs = text_splitter.split_documents(documents)
    
#     texts = [doc.page_content for doc in docs]
#     metadatas = [doc.metadata for doc in docs]

#     # Embed texts
#     vectors = embeddings.embed_documents(texts)

#     # Prepare for upsert
#     to_upsert = [
#         {
#             "id": f"{pdf_path.split('/')[-1]}-{i}",
#             "values": vectors[i],
#             "metadata": metadatas[i]
#         }
#         for i in range(len(texts))
#     ]

#     # Upload to Pinecone
#     index.upsert(
#         vectors=to_upsert,
#         namespace=pdf_path.split("/")[-1].replace(".pdf", "")
#     )
    
#     print(f"✅ Uploaded {pdf_path} to Pinecone!")

# print("🎯 All PDFs processed and stored in Pinecone.")









from pinecone import (
    Pinecone,
    ServerlessSpec
)
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
import os
import time
import re

# Initialize Pinecone client
os.environ['PINECONE_API_KEY'] = "pcsk_JbTgY_8nL8BcQhif14xgerADD1WhLgbJLbZnPzZ5KzsNkfnWYPH6nkaZMFRajhjGs38bg"
pc = Pinecone(api_key=os.environ['PINECONE_API_KEY'])

index_name = "desi-design-index"

# Create index if it doesn't exist
if index_name not in [index.name for index in pc.list_indexes()]:
    pc.create_index(
        name=index_name,
        dimension=384,  # for sentence-transformers/all-MiniLM-L6-v2
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )

# Connect to existing index
index = pc.Index(index_name)

# Initialize HuggingFace embedding model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"}  # Use "cuda" if you have GPU
)

# Text splitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=200
)

# Include multiple PDFs
pdf_paths = [
    "Data/systemdesign1.pdf",
    "Data/systemdesign2.pdf",
    "Data/systemdesign3.pdf"
]

def clean_text(text):
    """Clean text to make it suitable for embedding"""
    if not isinstance(text, str):
        return ""
    
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove any non-printable characters
    text = ''.join(c for c in text if c.isprintable() or c.isspace())
    
    # Remove text that's too short (likely not meaningful)
    if len(text.strip()) < 10:
        return ""
    
    return text.strip()

for pdf_path in pdf_paths:
    try:
        print(f"\n🔍 Processing {pdf_path}...")
        
        # Check if file exists
        if not os.path.exists(pdf_path):
            print(f"❌ File not found: {pdf_path}. Skipping...")
            continue
        
        # Create a namespace for this PDF
        namespace = pdf_path.split("/")[-1].replace(".pdf", "").lower()
        
        # Load and split the document
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        docs = text_splitter.split_documents(documents)
        
        # Extract and clean texts
        texts = []
        metadatas = []
        
        for doc in docs:
            cleaned_text = clean_text(doc.page_content)
            if cleaned_text:  # Only include non-empty texts
                texts.append(cleaned_text)
                metadatas.append(doc.metadata)
        
        print(f"Total valid chunks to process: {len(texts)}")
        
        # Process in smaller batches
        batch_size = 30  # Smaller batch size for more stability
        successful_chunks = 0
        failed_chunks = 0
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            batch_metadatas = metadatas[i:i+batch_size]
            
            print(f"Processing batch {i//batch_size + 1}/{(len(texts) + batch_size - 1)//batch_size}")
            
            # Extra checking and cleaning for this batch
            valid_batch_texts = []
            valid_batch_metadatas = []
            
            for j, text in enumerate(batch_texts):
                # Additional validation for each text in the batch
                if text and len(text.strip()) > 0:
                    valid_batch_texts.append(text)
                    valid_batch_metadatas.append(batch_metadatas[j])
            
            if not valid_batch_texts:
                print("Skipping empty batch")
                continue
                
            # Embed texts with robust error handling
            try:
                # Process one text at a time if there are issues
                all_vectors = []
                failed_indices = []
                
                for j, text in enumerate(valid_batch_texts):
                    try:
                        # Check if the text is valid for embedding
                        if not text or not isinstance(text, str) or len(text.strip()) == 0:
                            failed_indices.append(j)
                            continue
                            
                        # Try to embed each text individually
                        vector = embeddings.embed_query(text)
                        all_vectors.append(vector)
                        
                    except Exception as e:
                        print(f"⚠️ Failed to embed text at index {i+j}: {str(e)[:100]}...")
                        failed_indices.append(j)
                
                # Remove failed texts from the batch
                for index in sorted(failed_indices, reverse=True):
                    valid_batch_metadatas.pop(index)
                    valid_batch_texts.pop(index)
                
                # Only proceed if we have successfully embedded texts
                if all_vectors:
                    # Prepare for upsert
                    to_upsert = [
                        {
                            "id": f"{namespace}-{i+j}",
                            "values": all_vectors[j],
                            "metadata": {
                                **valid_batch_metadatas[j],
                                "text": valid_batch_texts[j][:100] + "..." if len(valid_batch_texts[j]) > 100 else valid_batch_texts[j]
                            }
                        }
                        for j in range(len(all_vectors))
                    ]
                    
                    # Upload to Pinecone
                    index.upsert(
                        vectors=to_upsert,
                        namespace=namespace
                    )
                    
                    successful_chunks += len(all_vectors)
                    print(f"✅ Successfully uploaded {len(all_vectors)} vectors in batch {i//batch_size + 1}")
                
                # Add a small delay to avoid rate limits
                time.sleep(0.5)
                
            except Exception as e:
                failed_chunks += len(valid_batch_texts)
                print(f"❌ Error processing batch {i//batch_size + 1}: {str(e)[:200]}...")
                print("Continuing with next batch...")
        
        print(f"📊 {pdf_path} summary: {successful_chunks} chunks embedded successfully, {failed_chunks} chunks failed")
    
    except Exception as e:
        print(f"❌ Failed to process {pdf_path}: {str(e)}")

print("\n🎯 All PDFs processed and stored in Pinecone.")