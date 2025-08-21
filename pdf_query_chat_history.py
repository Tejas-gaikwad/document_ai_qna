import os
import pdfplumber
import openai
import faiss
import numpy as np
import pickle
from dotenv import load_dotenv
from openai import OpenAI

# Load API key
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai.api_key = OPENAI_API_KEY
client = OpenAI()

# ---------------- PDF PROCESSING ----------------
def extract_text_from_pdf(pdf_path):
    """Extracts all text from a PDF file"""
    with pdfplumber.open(pdf_path) as pdf:
        text = ""
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text

def split_text_into_chunks(text, chunk_size=1500):
    """Splits text into overlapping chunks"""
    paragraphs = text.split("\n")
    chunks, current_chunk = [], ""
    for para in paragraphs:
        if len(current_chunk) + len(para) > chunk_size:
            chunks.append(current_chunk)
            current_chunk = para
        else:
            current_chunk += "\n" + para
    if current_chunk:
        chunks.append(current_chunk)
    return chunks

# ---------------- EMBEDDINGS ----------------
def get_batch_embeddings(chunks, batch_size=50):
    """Generates embeddings in batches to avoid token limits"""
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

# ---------------- FAISS ----------------
def build_faiss_index(text_chunks, dimension=1536):
    """Builds a FAISS index from text chunks"""
    embeddings = get_batch_embeddings(text_chunks, batch_size=50)
    embeddings = np.array(embeddings).astype('float32')

    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index

def query_index(query, index, text_chunks, k=3):
    """Searches FAISS index for relevant chunks"""
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=query
    )
    query_embedding = np.array(response.data[0].embedding).reshape(1, -1).astype('float32')
    distances, indices = index.search(query_embedding, k)
    results = [text_chunks[i] for i in indices[0]]
    return results

# ---------------- GPT-4 QA ----------------
def ask_gpt4(question, relevant_text, chat_history=None):
    """Asks GPT-4 using context + optional chat history"""
    context = "\n".join(relevant_text)
    prompt = f"Question: {question}\n\nContext:\n{context}\n\nAnswer:"

    messages = [{"role": "system", "content": "You are a helpful assistant that answers based on provided context."}]
    
    # include chat history if available
    if chat_history:
        messages.extend(chat_history)

    messages.append({"role": "user", "content": prompt})

    response = client.chat.completions.create(
        model="gpt-4",
        messages=messages,
        max_tokens=500,
        temperature=0.7,
    )

    answer = response.choices[0].message.content.strip()

    # update history
    if chat_history is not None:
        chat_history.append({"role": "user", "content": question})
        chat_history.append({"role": "assistant", "content": answer})

    return answer

# ---------------- MAIN PIPELINE ----------------
def run_scraper_and_query(pdf_path, query, index_file="book_index.faiss", chunks_file="text_chunks.pkl", chat_history=None):
    dimension = 1536

    # Load existing index
    if os.path.exists(index_file) and os.path.exists(chunks_file):
        print("Loading saved index and chunks...")
        index = faiss.read_index(index_file)
        with open(chunks_file, "rb") as f:
            text_chunks = pickle.load(f)
    else:
        print("Creating new index from PDF...")
        text = extract_text_from_pdf(pdf_path)
        text_chunks = split_text_into_chunks(text)

        index = build_faiss_index(text_chunks, dimension=dimension)

        # Save for reuse
        faiss.write_index(index, index_file)
        with open(chunks_file, "wb") as f:
            pickle.dump(text_chunks, f)

    # Search & Answer
    relevant_text = query_index(query, index, text_chunks)
    answer = ask_gpt4(query, relevant_text, chat_history=chat_history)
    return answer

# ---------------- RUN ----------------
if __name__ == "__main__":
    pdf_file = "./Bhagavad-gita_As_It_Is.pdf"
    chat_history = []  # keeps memory of conversation

    while True:
        query = input("\nAsk a question (or type 'exit'): ")
        if query.lower() == "exit":
            break
        answer = run_scraper_and_query(pdf_file, query, chat_history=chat_history)
        print("\nAnswer:", answer)
