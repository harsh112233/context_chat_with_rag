import streamlit as st
import tempfile
import os
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.memory import ChatMemoryBuffer

# Load environment variables
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
if not OPENAI_API_KEY:
    st.error("OPENAI_API_KEY not found in environment variables. Please set it in your secrets.toml file.")
    st.stop()

# App header and description
st.header("Context remembering AI chat with RAG")
st.write("Upload a PDF document and query the knowledge it contains. This app uses the document content and LLM-generated information to answer your queries. Supported file types: PDF file.")

# Initialize session state
if "chat_engine" not in st.session_state:
    st.session_state["chat_engine"] = None
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []
if "uploaded_file_content" not in st.session_state:
    st.session_state["uploaded_file_content"] = None
if "uploaded_file_index" not in st.session_state:
    st.session_state["uploaded_file_index"] = None
if 'documents' not in st.session_state:
    st.session_state.documents = None
if "button_disabled" not in st.session_state:
    st.session_state.button_disabled = True
if "user_message" not in st.session_state:
    st.session_state.user_message = None
if "previous_file" not in st.session_state:
    st.session_state.previous_file = None

# File upload
uploaded_file = st.file_uploader("**Upload a file**")

# Check if a new file is uploaded or the previous one is deleted
if uploaded_file != st.session_state.previous_file:
    if st.session_state.previous_file is not None:
        # Reset chat engine and history if the file changes
        st.session_state["chat_engine"] = None
        st.session_state["chat_history"] = []
        st.session_state.documents = None
        st.session_state.button_disabled = True
        st.success("Previous file removed. Chat engine and history reset.")
    st.session_state.previous_file = uploaded_file

if uploaded_file is not None and st.session_state.documents is None:
    try:
        # Create a temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            # Save the uploaded file to the temp directory
            file_path = os.path.join(temp_dir, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.read())  # Write bytes to disk
            # Load with SimpleDirectoryReader
            documents = SimpleDirectoryReader(temp_dir).load_data()
            st.session_state.documents = documents  # document loaded now
            st.success(f"File '{uploaded_file.name}' uploaded and loaded successfully!")
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")

# Function to initialize chat engine
def init_chat_engine():
    try:
        documents = st.session_state.documents
        index = VectorStoreIndex.from_documents(documents, show_progress=True)
        memory = ChatMemoryBuffer.from_defaults(token_limit=3000)
        st.success(f"File '{uploaded_file.name}' uploaded and loaded successfully!")
        chat_engine = index.as_chat_engine(
            chat_mode="context",
            memory=memory,
            system_prompt=(
                "You are a helpful and knowledgeable support agent. "
                "Your goal is to assist users with their inquiries, "
                "provide accurate information, and resolve issues in a friendly and professional manner."
            ),
        )
        return chat_engine
    except Exception as e:
        st.error(f"Failed to initialize chat engine: {e}")
        return None

# Display disabled button:
if uploaded_file is not None:
    st.session_state.button_disabled = False

# Load chat engine on button click
if st.button("Load Documents and Initialize Chat", disabled=st.session_state.button_disabled, key="load_button"):
    with st.spinner("Indexing documents..."):
        st.session_state["chat_engine"] = init_chat_engine()
        if st.session_state["chat_engine"]:
            st.success("Chat engine initialized successfully!")
        else:
            st.error("Failed to load chat engine. Check your data folder or API key.")

# Container
container = st.container(border=True)

# Chat interface
if st.session_state["chat_engine"]:
    container.subheader("Chat with the AI")
    user_query = st.chat_input(placeholder="Your message", key="user_input")

    # Handle user input
    if user_query:
        if user_query.lower() == "exit":
            st.session_state["chat_history"].append({"role": "user", "content": "Exit"})
            st.session_state["chat_history"].append({"role": "assistant", "content": "Goodbye! Chat ended."})
            st.session_state["chat_engine"] = None  # Reset chat engine
        else:
            try:
                with st.spinner("Generating response..."):
                    response = st.session_state["chat_engine"].chat(user_query)
                    st.session_state["chat_history"].append({"role": "user", "content": user_query})
                    st.session_state["chat_history"].append({"role": "assistant", "content": str(response)})
            except Exception as e:
                st.error(f"Error generating response: {e}")

    # Display chat history
    for message in st.session_state["chat_history"]:
        if message["role"] == "user":
            container.write(f"**You:** {message['content']}")
        else:
            container.write(f"**AI:** {message['content']}")

# Instructions
st.sidebar.markdown("""
### Instructions
1. Upload any PDF file using the file uploader.
2. Click 'Load Documents and Initialize Chat' to start.
3. Type your question and press Enter.
4. Type 'exit' to end the chat.
5. Delete and upload a new file to reset the chat.
""")