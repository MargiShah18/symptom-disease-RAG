# app.py

import os
from PyPDF2 import PdfReader
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema import SystemMessage, HumanMessage
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import Pinecone as LangchainPinecone
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# Load environment variables
load_dotenv()
OPENAI_API_KEY2 = os.getenv("OPENAI_API_KEY2")
PINECONE_API_KEY2 = os.getenv("PINECONE_API_KEY2")

if not OPENAI_API_KEY2 or not PINECONE_API_KEY2:
    raise ValueError("Missing API Keys. Please check your .env file.")

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

chat = ChatOpenAI(openai_api_key=OPENAI_API_KEY2, model='gpt-3.5-turbo')
pc = Pinecone(api_key=PINECONE_API_KEY2)
index_name = "rag-chatbot-index"

if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=1536,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-west-2")
    )
index = pc.Index(index_name)

def clear_pinecone():
    try:
        index.delete(delete_all=True)
    except Exception as e:
        print(f"Error clearing Pinecone: {e}")

def extract_text_from_pdf(pdf_path):
    try:
        text = ""
        with open(pdf_path, 'rb') as file:
            reader = PdfReader(file)
            for page in reader.pages:
                text += page.extract_text() + "\n"
        return text
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
        return None

def process_and_store_file(file_path):
    try:
        if file_path.lower().endswith('.pdf'):
            text = extract_text_from_pdf(file_path)
            if text:
                store_in_pinecone(text, os.path.basename(file_path))
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")

def store_in_pinecone(text, filename):
    try:
        embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY2)
        vectors = embeddings.embed_documents([text])
        file_id = str(hash(filename))[:8]
        index.upsert(vectors=[{
            'id': file_id,
            'values': vectors[0],
            'metadata': {'text': text, 'filename': filename}
        }])
    except Exception as e:
        print(f"Error storing in Pinecone: {e}")

def delete_file_from_pinecone(file_id):
    try:
        index.delete(ids=[file_id])
    except Exception as e:
        print(f"Error deleting from Pinecone: {e}")

class FileHandler(FileSystemEventHandler):
    def on_created(self, event):
        if not event.is_directory and event.src_path.lower().endswith('.pdf'):
            process_and_store_file(event.src_path)

    def on_deleted(self, event):
        if not event.is_directory and event.src_path.lower().endswith('.pdf'):
            filename = os.path.basename(event.src_path)
            file_id = str(hash(filename))[:8]
            delete_file_from_pinecone(file_id)

    def on_modified(self, event):
        if not event.is_directory and event.src_path.lower().endswith('.pdf'):
            filename = os.path.basename(event.src_path)
            file_id = str(hash(filename))[:8]
            delete_file_from_pinecone(file_id)
            process_and_store_file(event.src_path)

def setup_file_watcher(UPLOAD_FOLDER):
    event_handler = FileHandler()
    observer = Observer()
    observer.schedule(event_handler, UPLOAD_FOLDER, recursive=False)
    observer.start()
    return observer

def process_existing_files(UPLOAD_FOLDER):
    clear_pinecone()
    for filename in os.listdir(UPLOAD_FOLDER):
        if filename.lower().endswith('.pdf'):
            file_path = os.path.join(UPLOAD_FOLDER, filename)
            if os.path.isfile(file_path):
                process_and_store_file(file_path)

def augment_prompt(query):
    try:
        # Simple greeting detection
        greetings = ['hi', 'hello', 'hey', 'hola']
        if query.strip().lower() in greetings:
            return "The user greeted the assistant. Respond warmly and ask how you can help."

        embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY2)
        query_vector = embeddings.embed_query(query)
        results = index.query(vector=query_vector, top_k=3, include_metadata=True)
        context = "\n\n".join(match.metadata['text'] for match in results.matches)

        return f"""Context from your documents:
{context}

User's question: {query}

Please answer the question based on the provided context. If the context doesn't contain relevant information, say "I don't have enough information to answer that question." """
    except Exception as e:
        print(f"Error augmenting prompt: {e}")
        return query


def generate_response(augmented_query):
    try:
        prompt = HumanMessage(content=augmented_query)
        res = chat.invoke([SystemMessage(content="You are a helpful assistant."), prompt])
        return res.content
    except Exception as e:
        print(f"Error generating response: {e}")
        return None
