import os
import re
import streamlit as st
import base64
import uuid
import tempfile
import faiss
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from sentence_transformers import SentenceTransformer
from PyPDF2 import PdfReader
from document_uploader import build_indexes
from global_settings import LANGUAGE_MODEL, EMBEDDING_MODEL, MAX_NEW_TOKENS, MAX_LENGTH, TEMPERATURE, TOP_K, TOP_P
from logging_functions import log_action

# Initialize the embedding model
embedding_model = SentenceTransformer(EMBEDDING_MODEL)

# Load the tokenizer and language model
tokenizer = AutoTokenizer.from_pretrained(LANGUAGE_MODEL)
model = AutoModelForCausalLM.from_pretrained(LANGUAGE_MODEL)

# Create a text generation pipeline
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=MAX_NEW_TOKENS,
    max_length=MAX_LENGTH,
    truncation=True,
    temperature=TEMPERATURE,
    top_k=TOP_K,
    top_p=TOP_P
)

if "id" not in st.session_state:
    st.session_state.id = uuid.uuid4()
    st.session_state.file_cache = {}
    st.session_state.messages = []
    st.session_state.document_data = {}

session_id = st.session_state.id

def load_pdf(file_path):
    reader = PdfReader(file_path)
    text = []
    for page_number, page in enumerate(reader.pages, start=1):
        page_text = page.extract_text() or ""  # Ensure no None is added
        text.append((page_text, page_number))
    return text

def display_pdf(file):
    base64_pdf = base64.b64encode(file.read()).decode("utf-8")
    pdf_display = f"""<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="100%" type="application/pdf"></iframe>"""
    st.sidebar.markdown(pdf_display, unsafe_allow_html=True)

def generate_embeddings(text):
    if isinstance(text, list):
        return embedding_model.encode(text)
    else:
        return embedding_model.encode([text])

def generate_response(prompt, index, sentences, pipe):
    # Embed the prompt and perform a similarity search
    prompt_embedding = embedding_model.encode([prompt])
    D, I = index.search(prompt_embedding, k=TOP_K)  # Retrieve top K chunks
    results = [sentences[i] for i in I[0]]

    # Check if any relevant chunks were found
    if not results:
        print("No relevant chunks found for the query.")
        return "I couldn't find any information related to your query in the document."

    # Construct the context
    context = "\n".join([f"### Page {index}:\n{chunk}\n" for chunk, index in results])

    # Craft the final prompt
    final_prompt = f"""
    {context}

    Question: {prompt}

    Answer: 
    """

    # Generate the response
    response = pipe(final_prompt, max_new_tokens=MAX_NEW_TOKENS, max_length=MAX_LENGTH, truncation=True)[0][
        "generated_text"]
    # Print the full model output for debugging
    print("Full Model Output:", response)
    response_list = response.strip().split(". ")[:2]  # Limit to first two sentences
    page_number = results[0][1]

    # Format the response as dot points
    formatted_response = "\n".join([f"{sentence}" for sentence in response_list])

    return formatted_response
    # # Print the full model output for debugging
    # print("Full Model Output:", response)
    #
    # # Extract the answer and source(s) using regex
    # match = re.search(r"(.*?)\s*(\((Page(?:s)?\s[\d,\s]+)\))?", response, re.DOTALL)
    # if match:
    #     answer = match.group(1).strip()
    #     source = match.group(2).strip() if match.group(2) else ""
    #     return f"{answer} {source}"
    # else:
    #     # If no match is found, return a generic message
    #     return "I'm still learning and trying my best to answer your questions. Could you please try rephrasing or asking a different question?"

    # Craft the final prompt with even more explicit instructions
    # final_prompt = f"""
    #     {context}
    #
    #     **Instructions:**
    #
    #     * You MUST provide a response to the question based on the context above.
    #     * If you can confidently answer based on the context, cite the page number(s) where you found the information, e.g., (Page 1).
    #     * If you're unsure or the answer isn't explicitly stated, provide your best guess based on the context and your general knowledge, but indicate that you're not completely certain.
    #     * Keep your response concise, clear, and focused on answering the question.
    #     * Do not repeat the question or include any part of the instructions in your response.
    #
    #     **Question:** {prompt}
    #
    #     **Answer:**
    #     """

with st.sidebar:
    st.header("Add your PDF!")
    uploaded_file = st.file_uploader("Choose your `.pdf` file", type="pdf")

    if uploaded_file:
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                file_path = os.path.join(temp_dir, uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getvalue())

                st.write("Indexing your document...")
                document_data = load_pdf(file_path)
                text = [data[0] for data in document_data]
                embeddings = embedding_model.encode(text)

                vector_index = build_indexes(embeddings)

                st.success("Ready to Chat!")
                display_pdf(uploaded_file)

                st.session_state.index = vector_index
                st.session_state.sentences = document_data
                st.session_state.document_type = 'pdf'

                # Log the document upload action
                log_action(f"Uploaded document: {uploaded_file.name}", "Document Upload")

        except Exception as e:
            st.error(f"An error occurred: {e}")

            # Log the error
            log_action(f"Error uploading document: {e}", "Error")

            st.stop()

st.title("Local Chatbot")
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask a question!"):
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        response = generate_response(prompt, st.session_state.index, st.session_state.sentences, pipe)

        message_placeholder.markdown(response.strip())

        st.session_state.messages.append({"role": "assistant", "content": response.strip()})

        # Log the user's question and the assistant's response
        log_action(f"User question: {prompt}", "User Interaction")
        log_action(f"Assistant response: {response.strip()}", "Assistant Response")