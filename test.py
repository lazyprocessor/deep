import streamlit as st
import uuid
# import os
import requests
# Remove this since you won't need a hard-coded import anymore:
# from config import pinecone_key
from openai import OpenAI
from pinecone import Pinecone
import PyPDF2
import re
import docx2txt
from sentence_transformers import SentenceTransformer

# We no longer define PINECONE_API_KEY = pinecone_key, since it will come from the user

@st.cache_resource
def init():
    print("Loading model and connecting to Pinecone only once...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # Now we get Pinecone key from st.session_state
    pinecone_api_key = st.session_state["pinecone_api_key"]
    pc = Pinecone(api_key=pinecone_api_key)
    INDEX_NAME = "testing"  # Change to your actual index name
    index = pc.Index(INDEX_NAME)

    return model, index

def extract_text_from_pdf(file) -> str:
    pdf_reader = PyPDF2.PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() + "\n"
    return text

def chunk_text(text, chunk_size=5000):
    text = text.replace("\n", " ")
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end
    return chunks

def get_embedding(text: str, sentence_model) -> list[float]:
    return sentence_model.encode(text, convert_to_numpy=True).tolist()

def extract_text_from_docx(docx_file):
    return docx2txt.process(docx_file)

def upsert_pdf_docx_chunks_to_pinecone(user_id, file, index, model):
    file_name = file.name.lower()
    # Decide how to extract text based on file extension
    if file_name.endswith(".pdf"):
        full_text = extract_text_from_pdf(file)
    elif file_name.endswith(".docx"):
        full_text = extract_text_from_docx(file)
    else:
        raise ValueError("Unsupported file format. Please upload a PDF or DOCX.")

    chunks = chunk_text(full_text, chunk_size=500)
    
    vectors_to_upsert = []
    for i, chunk in enumerate(chunks):
        print("Inserting Chunk : ",chunk)
        embedding = get_embedding(chunk, model)
        vector_id = f"{user_id}_{i}"
        metadata = {"user_id": user_id, "chunk_index": i, "text": chunk}
        vectors_to_upsert.append((vector_id, embedding, metadata))
    
    index.upsert(vectors=vectors_to_upsert)

def query_pinecone(user_id, query_text, index, model, top_k=3):
    print("Querying Pinecone with user_id : ",user_id)
    query_embedding = get_embedding(query_text, model)
    result = index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True,
        filter={"user_id": {"$eq": user_id}}
    )
    return result

def create_context(matches):
    if not matches:
        return ""
    context_texts = [m['metadata']['text'] for m in matches]
    return "\n\n".join(context_texts)

def ask_openai_with_context(user_query, context, chat_history, openai_api_key):
    system_prompt = f"""
    "You are a helpful and responsible AI assistant. Answer the user’s query based on the provided document content and chat history."
    Use the following context and past exchanged messages to provide responses.
    If user asked try to make summary  from past messages or chat history then use previous interactions  for that.
    If the answer is available in the document or chat history, provide it. Otherwise, state that the information is not available.
    Do not assume or generate answers beyond the given content.
    If a user asks for a summary, generate a concise overview based on past interactions.
    If a user requests illegal, unethical, or harmful content, respond with: ‘I’m sorry, but I can’t provide that information.’
    Ensure that responses remain safe, unbiased, and factual.
    Context : (
    {context})
    summarize the past all history if user ask.
    """

    client = OpenAI(api_key=openai_api_key)
    messages = [{"role": "system", "content": system_prompt}]

    
    recent_messages = []

    for chat in reversed(chat_history):  # Reverse to get the most recent first
        recent_messages.append({"role": "assistant", "content": chat["response"]})
        recent_messages.append({"role": "user", "content": chat["query"]})

        if len(recent_messages) >= 10:  
            break

    # Add recent messages in the correct order
    messages.extend(reversed(recent_messages))  # Reverse again to maintain order

    # Add the latest user query
    messages.append({"role": "user", "content": user_query})

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0.3
    )
    return response.choices[0].message.content

def chat_init(user_query, user_id, index, model, openai_api_key):
    pinecone_results = query_pinecone(user_id, user_query, index, model, top_k=3)
    print("Pinecone result : ", pinecone_results)
    top_chunks = pinecone_results.get("matches", [])
    context = create_context(top_chunks)
    chat_history = st.session_state["chat_history"]
    print("Chat history : ",chat_history)
    print("Context : ",context)
    answer = ask_openai_with_context(user_query, context, chat_history, openai_api_key)
    return answer

def main():
    st.title("PDF Chatbot with Pinecone & OpenAI")
    st.title(" OpenAI API Key Validator")
    if "openai_api_key" not in st.session_state:
        st.subheader("Enter your OpenAI API key to begin")
        user_key = st.text_input("OpenAI API Key", type="password")
    
        if st.button("Submit OpenAI Key"):
            if user_key.strip():
                headers = {"Authorization": f"Bearer {user_key.strip()}"}
                response = requests.get("https://api.openai.com/v1/models", headers=headers)

                if response.status_code == 200:
                    st.session_state["openai_api_key"] = user_key.strip()
                    st.success("API Key is valid! 🎉")
                    st.rerun()
                else:
                    st.error("Invalid OpenAI API key. Please check and try again.")
            else:
                 st.warning("Please provide a valid OpenAI API Key.")
        st.stop()
    else:
         st.success("API Key already set! You can proceed. 🚀")

    st.title("Pinecone API Key Validator")

    if "pinecone_api_key" not in st.session_state:
        st.subheader("Enter your Pinecone API key to begin")
        pinecone_key_input = st.text_input("Pinecone API Key", type="password")

        if st.button("Submit Pinecone Key"):
            if pinecone_key_input.strip():
                st.session_state["pinecone_api_key"] = pinecone_key_input.strip()
                st.rerun()
            else:
                st.warning("Please provide a valid Pinecone API Key.")
        st.stop()

    # 3) Init Model & Index

    
    model, index = init()
    openai_api_key = st.session_state["openai_api_key"]

    # Assign or retrieve a unique user ID
    if "user_id" not in st.session_state:
        st.session_state["user_id"] = str(uuid.uuid4())
    user_id = st.session_state["user_id"]

    # Initialize chat history
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []
    if "file_processed" not in st.session_state:
        st.session_state["file_processed"] = False 

    st.write(f"Your user ID is: **{user_id}**")

        # PDF/DOCX Upload
    st.header("Upload Your PDF or DOCX")
    uploaded_file = st.file_uploader("Choose a PDF or DOCX file", type=["pdf", "docx"])
    
    # Add file size restriction (3 MB = 3 * 1024 * 1024 bytes)
    max_file_size = 3 * 1024 * 1024  # 3 MB in bytes
    
    if uploaded_file is not None:
        # Check file size before processing
        file_size = uploaded_file.size
        
        if file_size > max_file_size:
            st.error(f"File size exceeds the maximum limit of 3 MB. Your file is {file_size / (1024 * 1024):.2f} MB.")
        elif not st.session_state["file_processed"]:
            user_id = st.session_state["user_id"]
            # 'index' is your Pinecone index, 'model' is your chosen embedding model
            # upsert to Pinecone
            upsert_pdf_docx_chunks_to_pinecone(user_id, uploaded_file, index, model)
            st.success("Your file has been indexed successfully!")
            st.session_state["file_processed"] = True

    # Display chat history 
    st.header("Chat History")
    for chat in st.session_state["chat_history"]:
        st.markdown(f"**You:** {chat['query']}")
        st.markdown(f"**Bot:** {chat['response']}")
        st.markdown("---")

    # Chat Interface
    st.header("Ask a Question About Your PDF")
    user_query = st.text_input("Type your question here...")

    if st.button("Send"):
        if user_query.strip():
            answer = chat_init(user_query, user_id, index, model, openai_api_key)
            st.session_state["chat_history"].append(
                {"query": user_query, "response": answer}
            )
            st.rerun()
        else:
            st.warning("Please enter a question first.")

if __name__ == "__main__":
    main()