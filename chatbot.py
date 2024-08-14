import os
import streamlit as st
import base64
import uuid
import tempfile
from transformers import pipeline
from sentence_transformers import SentenceTransformer
import faiss
from PyPDF2 import PdfReader
from document_uploader import build_indexes, load_index, ingest_documents

if "id" not in st.session_state:
    st.session_state.id = uuid.uuid4()
    st.session_state.file_cache = {}
    st.session_state.messages = []  # Initialize messages list in session state

session_id = st.session_state.id


def display_pdf(file):
    st.markdown("### PDF Preview")
    base64_pdf = base64.b64encode(file.read()).decode("utf-8")
    pdf_display = f"""<iframe src="data:application/pdf;base64,{base64_pdf}" width="400" height="100%" type="application/pdf"></iframe>"""
    st.markdown(pdf_display, unsafe_allow_html=True)


def load_pdf(file_path):
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text


# Initialize the local embeddings model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

with st.sidebar:
    st.header("Add your documents!")
    uploaded_file = st.file_uploader("Choose your `.pdf` file", type="pdf")

    if uploaded_file:
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                file_path = os.path.join(temp_dir, uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getvalue())

                st.write("Indexing your document...")
                text = load_pdf(file_path)
                sentences = text.split("\n")  # Simple sentence splitting
                embeddings = embedding_model.encode(sentences)

                # Initialize FAISS and build the index
                vector_index = build_indexes(embeddings)

                st.success("Ready to Chat!")
                display_pdf(uploaded_file)

                # Store the index in session state
                st.session_state.index = vector_index
                st.session_state.sentences = sentences

        except Exception as e:
            st.error(f"An error occurred: {e}")
            st.stop()

st.title("Local Chatbot")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Streamlit chat UI
# If user inputs a new question
if prompt := st.chat_input("Ask a question!"):
    # Add user question to the session state messages
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    # Chatbot processes the question and provides a response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        # Embed the prompt and perform a similarity search
        prompt_embedding = embedding_model.encode([prompt])
        D, I = st.session_state.index.search(prompt_embedding, k=3)
        results = [st.session_state.sentences[i] for i in I[0]]

        # Concatenate the most similar sentences to form the response
        response = " ".join(results)

        # Stream the response in the chat
        for chunk in response.split(" "):
            full_response += chunk + " "
            message_placeholder.markdown(full_response + "▌")
        message_placeholder.markdown(full_response)

        # Add Chatbot response to the session state messages
        st.session_state.messages.append({"role": "assistant", "content": full_response})


