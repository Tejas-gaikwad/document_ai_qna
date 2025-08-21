import os
import pdfplumber
import openai
import faiss
import numpy as np
import pickle
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai.api_key = OPENAI_API_KEY
client = OpenAI()

# Function to extract text from a PDF
def extract_text_from_pdf(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        text = ""
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text

# Get embeddings for a batch of chunks
def get_batch_embeddings(chunks, batch_size=50):
    embeddings = []
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i+batch_size]
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=batch
        )
        batch_embeddings = [data.embedding for data in response.data]
        embeddings.extend(batch_embeddings)
        print(f"Processed {i+len(batch)}/{len(chunks)} chunks")
    return embeddings

# Split text into chunks
def split_text_into_chunks(text, chunk_size=1500):
    paragraphs = text.split("\n")
    chunks = []
    current_chunk = ""
    for para in paragraphs:
        if len(current_chunk) + len(para) > chunk_size:
            chunks.append(current_chunk)
            current_chunk = para
        else:
            current_chunk += "\n" + para
    if current_chunk:
        chunks.append(current_chunk)
    return chunks

# Build FAISS index
def index_text(text_chunks, index):
    embeddings = get_batch_embeddings(text_chunks, batch_size=50)
    embeddings = np.array(embeddings).astype('float32')
    index.add(embeddings)

# Query FAISS index
def query_index(query, index, text_chunks, k=3):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=query
    )
    query_embedding = np.array(response.data[0].embedding).reshape(1, -1).astype('float32')
    distances, indices = index.search(query_embedding, k)
    results = [text_chunks[i] for i in indices[0]]
    return results

# Ask GPT-4 with retrieved context
def ask_gpt4(question, relevant_text):
    context = "\n".join(relevant_text)
    prompt = f"Question: {question}\n\nContext:\n{context}\n\nAnswer:"
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that answers based on provided context."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=500,
        temperature=0.7,
    )
    return response.choices[0].message.content.strip()

# Main pipeline with persistence
def run_scraper_and_query(pdf_path, query, index_file="book_index.faiss", chunks_file="text_chunks.pkl"):
    dimension = 1536

    # If saved index exists, load it
    if os.path.exists(index_file) and os.path.exists(chunks_file):
        print("Loading saved index and chunks...")
        index = faiss.read_index(index_file)
        with open(chunks_file, "rb") as f:
            text_chunks = pickle.load(f)
    else:
        print("Creating new index from PDF...")
        text = extract_text_from_pdf(pdf_path)
        text_chunks = split_text_into_chunks(text)

        index = faiss.IndexFlatL2(dimension)
        index_text(text_chunks, index)

        # Save index & chunks
        faiss.write_index(index, index_file)
        with open(chunks_file, "wb") as f:
            pickle.dump(text_chunks, f)

    # Query the index
    relevant_text = query_index(query, index, text_chunks)
    answer = ask_gpt4(query, relevant_text)
    return answer

# Run
pdf_file = "./Bhagavad-gita_As_It_Is.pdf"
if __name__ == "__main__":
    pdf_path = pdf_file
    query = "who is krsna" 
    # who are first 10 charcters of dissciplic succession
    answer = run_scraper_and_query(pdf_path, query)
    print("Answer:", answer)
