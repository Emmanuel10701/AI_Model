import streamlit as st
import tempfile
import os
import time
from langchain_community.document_loaders import PyPDFLoader, CSVLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain_community.embeddings import HuggingFaceEmbeddings
import requests
import json
import re
import openai 

# Try to import docx2txt, but handle if it's not available
try:
    from langchain_community.document_loaders import Docx2txtLoader
    DOCX_SUPPORT = True
except ImportError:
    DOCX_SUPPORT = False
    st.warning("DOCX support is not available. Please install docx2txt: pip install docx2txt")

# Set page config with modern theme
st.set_page_config(
    page_title="Multi-AI Document Assistant", 
    layout="wide",
    page_icon="📄",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        color: #1f77b4;
        margin-bottom: 0.5rem;
        font-weight: 700;
    }
    .subheader {
        font-size: 1.1rem;
        color: #4a4a4a;
        margin-bottom: 1.5rem;
    }
    .chat-container {
        background-color: #f8f9fa;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
        height: calc(100vh - 300px);
        overflow-y: auto;
    }
    .user-message {
        background-color: #e3f2fd;
        padding: 16px 20px;
        border-radius: 18px;
        margin: 12px 0;
        border-left: 4px solid #2196f3;
        max-width: 80%;
        margin-left: auto;
    }
    .assistant-message {
        background-color: #f1f8e9;
        padding: 16px 20px;
        border-radius: 18px;
        margin: 12px 0;
        border-left: 4px solid #4caf50;
        max-width: 80%;
        margin-right: auto;
    }
    .message-time {
        font-size: 0.7rem;
        color: #777;
        margin-top: 8px;
        text-align: right;
    }
    .stButton button {
        width: 100%;
        border-radius: 8px;
        background-color: #1f77b4;
        transition: all 0.3s;
    }
    .stButton button:hover {
        background-color: #1565c0;
        transform: translateY(-2px);
    }
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
    }
    .file-name {
        font-weight: 500;
        color: #1f77b4;
    }
    .streaming-response {
        min-height: 20px;
        line-height: 1.6;
    }
    .formatted-list {
        margin-left: 20px;
        margin-bottom: 12px;
    }
    .formatted-list li {
        margin-bottom: 8px;
        line-height: 1.5;
    }
    .topic-header {
        font-weight: 600;
        color: #1f77b4;
        margin-top: 15px;
        margin-bottom: 5px;
        font-size: 1.05rem;
    }
    .subtopic {
        font-weight: 500;
        color: #333;
        margin-top: 10px;
        margin-bottom: 5px;
    }
    .highlight-box {
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 12px 15px;
        border-radius: 8px;
        margin: 10px 0;
    }
    .numbered-list {
        margin-left: 20px;
        margin-bottom: 12px;
        list-style-type: decimal;
    }
    .numbered-list li {
        margin-bottom: 8px;
        line-height: 1.5;
    }
    .feature-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
        gap: 15px;
        margin: 15px 0;
    }
    .feature-card {
        background: white;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        border-left: 4px solid #4caf50;
    }
    .feature-title {
        font-weight: 600;
        color: #1f77b4;
        margin-bottom: 8px;
    }
    .feature-desc {
        font-size: 0.9rem;
        color: #555;
    }
    .model-selector {
        background-color: #f0f7ff;
        padding: 15px;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
        margin-bottom: 15px;
    }
    /* Hide Streamlit's default elements */
    header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "processed" not in st.session_state:
    st.session_state.processed = False
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "selected_model" not in st.session_state:
    st.session_state.selected_model = "Google Gemini"

# Initialize embeddings with proper error handling
@st.cache_resource
def load_embeddings():
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': False}
        )
        
        # Test the embeddings to make sure they work
        test_embeddings = embeddings.embed_documents(["test"])
        if len(test_embeddings) > 0 and len(test_embeddings[0]) > 0:
            return embeddings
        else:
            st.error("Embeddings returned empty results")
            return None
    except Exception as e:
        st.error(f"Error loading embedding model: {str(e)}")
        return None

# Load embeddings and handle None case
embeddings_result = load_embeddings()
if embeddings_result is None:
    st.error("Failed to load embeddings. Please check your installation.")
    st.stop()

embeddings = embeddings_result

# Function to format text with proper structure and formatting
def format_response(text):
    """Format text with proper structure, lists, and formatting"""
    # Remove unwanted asterisks
    text = re.sub(r'\*\*(?=\w)', '', text)  # Remove ** at start of words
    text = re.sub(r'\*\*$', '', text)  # Remove ** at end
    
    # Convert topic headers
    text = re.sub(r'(\w+.*?:)(?=\s)', r'<div class="topic-header">\1</div>', text)
    
    # Convert numbered items to HTML list
    text = re.sub(r'(\d+\.\s)(.*?)(?=\n\d+\.|\n\n|$)', r'<li>\2</li>', text, flags=re.DOTALL)
    
    # Wrap numbered lists in ol tags
    text = re.sub(r'(<li>.*?</li>)(?=\n[^<]|$)', r'<ol class="numbered-list">\1</ol>', text, flags=re.DOTALL)
    
    # Convert bullet points to HTML list
    text = re.sub(r'^\s*[\-\*]\s+(.*?)(?=\n[\-\*]|\n\n|$)', r'<li>\1</li>', text, flags=re.MULTILINE|re.DOTALL)
    
    # Wrap bullet lists in ul tags
    text = re.sub(r'(<li>.*?</li>)(?=\n[^<]|$)', r'<ul class="formatted-list">\1</ul>', text, flags=re.DOTALL)
    
    # Handle bold text with ** (clean version)
    text = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', text)
    
    # Add highlight boxes for important information
    text = re.sub(r'(Information you need to provide:|Your Details.*?:|Remember.*?:)', 
                 r'<div class="highlight-box">\1</div>', text, flags=re.IGNORECASE)
    
    # Handle line breaks
    text = text.replace('\n', '<br>')
    
    # Clean up any remaining asterisks
    text = text.replace('*', '')
    
    return text

# Function to simulate streaming response
def stream_response(text, speed=0.02):
    """Simulate streaming text response word by word"""
    words = text.split()
    placeholder = st.empty()
    partial_response = ""
    
    for word in words:
        partial_response += word + " "
        formatted_response = format_response(partial_response)
        placeholder.markdown(f'<div class="assistant-message"><strong>Assistant:</strong> {formatted_response}</div>', unsafe_allow_html=True)
        time.sleep(speed)
    
    return partial_response

# Function to query Google Gemini API
def query_gemini(prompt, context="", api_key=None, stream=False):
    if not api_key:
        return "Please enter your Google Gemini API key in the sidebar"
    
    try:
        # Gemini API endpoint
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key}"
        
        # Prepare the request payload
        payload = {
            "contents": [
                {
                    "parts": [
                        {
                            "text": f"Based on the following documents:\n\n{context}\n\nAnswer this question: {prompt}. Please structure your response with clear topics and subtopics, using bullet points for lists and bold text for important information. Avoid using asterisks (*) in your response."
                        }
                    ]
                }
            ],
            "generationConfig": {
                "temperature": 0.3,
                "maxOutputTokens": 1024
            }
        }
        
        # Make the API request
        headers = {'Content-Type': 'application/json'}
        response = requests.post(url, headers=headers, data=json.dumps(payload))
        
        if response.status_code == 200:
            result = response.json()
            full_response = result['candidates'][0]['content']['parts'][0]['text']
            
            if stream:
                return stream_response(full_response)
            else:
                return full_response
        else:
            return f"API Error: {response.status_code} - {response.text}"
            
    except Exception as e:
        return f"Error: {str(e)}"

# Add this import at the top of your file
from openai import OpenAI

def query_openai(prompt, context="", api_key=None, stream=False):
    if not api_key:
        return "Please enter your OpenAI API key in the sidebar"
    
    try:
        # For new OpenAI versions (v1.0.0+)
        client = OpenAI(api_key=api_key)

        # Prepare the messages
        messages = [
            {
                "role": "system",
                "content": (
                    f"You are a helpful assistant that answers questions based on the provided documents. "
                    f"Here is the context from the documents:\n\n{context}\n\n"
                    "Please structure your response with clear topics and subtopics, "
                    "using bullet points for lists and bold text for important information. "
                    "Avoid using asterisks (*) in your response."
                )
            },
            {
                "role": "user",
                "content": prompt
            }
        ]

        # Use non-streaming for simplicity
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            temperature=0.7,
            max_tokens=1024
        )

        content = response.choices[0].message.content
        
        # Simulate streaming if requested
        if stream:
            return stream_response(content)
        else:
            return content

    except Exception as e:
        return f"Error: {str(e)}"
        
        
def query_deepseek(prompt, context="", api_key=None, stream=False):
    if not api_key:
        return "Please enter your DeepSeek API key in the sidebar"
    
    try:
        # DeepSeek API endpoint
        url = "https://api.deepseek.com/v1/chat/completions"
        
        # Prepare the request payload
        payload = {
            "model": "deepseek-chat",
            "messages": [
                {
                    "role": "system",
                    "content": f"You are a helpful assistant that answers questions based on the provided documents. Here is the context from the documents:\n\n{context}\n\nPlease structure your response with clear topics and subtopics, using bullet points for lists and bold text for important information. Avoid using asterisks (*) in your response."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": 0.7,
            "max_tokens": 1024,
            "stream": False
        }
        
        # Make the API request
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {api_key}'
        }
        
        response = requests.post(url, headers=headers, data=json.dumps(payload))
        
        if response.status_code == 200:
            result = response.json()
            full_response = result['choices'][0]['message']['content']
            
            if stream:
                return stream_response(full_response)
            else:
                return full_response
        else:
            return f"API Error: {response.status_code} - {response.text}"
            
    except Exception as e:
        return f"Error: {str(e)}"

# Function to query LLaMA API (using Replicate)
def query_llama(prompt, context="", api_key=None, stream=False):
    if not api_key:
        return "Please enter your Replicate API key for LLaMA in the sidebar"
    
    try:
        # Replicate API endpoint for LLaMA
        url = "https://api.replicate.com/v1/predictions"
        
        # Prepare the request payload
        payload = {
            "version": "a16z-infra/llama7b-v2-chat:4f0a4744c7295c024a1de15e1a63c880d3da035fa1f49bfd344fe076074c8eea",
            "input": {
                "prompt": f"Based on the following documents:\n\n{context}\n\nAnswer this question: {prompt}. Please structure your response with clear topics and subtopics, using bullet points for lists and bold text for important information. Avoid using asterisks (*) in your response.",
                "max_length": 1024,
                "temperature": 0.7
            }
        }
        
        # Make the API request
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Token {api_key}'
        }
        
        response = requests.post(url, headers=headers, data=json.dumps(payload))
        
        if response.status_code == 200:
            result = response.json()
            prediction_id = result['id']
            
            # Wait for the prediction to complete
            prediction_url = f"https://api.replicate.com/v1/predictions/{prediction_id}"
            for _ in range(10):  # Try 10 times with delays
                time.sleep(1)
                prediction_response = requests.get(prediction_url, headers=headers)
                prediction_result = prediction_response.json()
                
                if prediction_result['status'] == 'succeeded':
                    full_response = ''.join(prediction_result['output'])
                    
                    if stream:
                        return stream_response(full_response)
                    else:
                        return full_response
                elif prediction_result['status'] == 'failed':
                    return "LLaMA API request failed"
            
            return "LLaMA API request timed out"
        else:
            return f"API Error: {response.status_code} - {response.text}"
            
    except Exception as e:
        return f"Error: {str(e)}"

# Document processing with better error handling and progress
def process_documents(uploaded_files):
    documents = []
    supported_files = []
    unsupported_files = []
    
    # Create progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, uploaded_file in enumerate(uploaded_files):
        try:
            # Update progress
            progress = (i + 1) / len(uploaded_files)
            progress_bar.progress(progress)
            status_text.text(f"Processing {uploaded_file.name}...")
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=uploaded_file.name) as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_file_path = tmp_file.name
                
            try:
                if uploaded_file.name.endswith('.pdf'):
                    loader = PyPDFLoader(tmp_file_path)
                    docs = loader.load()
                    documents.extend(docs)
                    supported_files.append(uploaded_file.name)
                    st.success(f"Processed PDF: {uploaded_file.name}")
                    
                elif uploaded_file.name.endswith('.csv'):
                    loader = CSVLoader(tmp_file_path)
                    docs = loader.load()
                    documents.extend(docs)
                    supported_files.append(uploaded_file.name)
                    st.success(f"Processed CSV: {uploaded_file.name}")
                    
                elif uploaded_file.name.endswith('.docx') and DOCX_SUPPORT:
                    loader = Docx2txtLoader(tmp_file_path)
                    docs = loader.load()
                    documents.extend(docs)
                    supported_files.append(uploaded_file.name)
                    st.success(f"Processed DOCX: {uploaded_file.name}")
                    
                elif uploaded_file.name.endswith('.txt'):
                    loader = TextLoader(tmp_file_path)
                    docs = loader.load()
                    documents.extend(docs)
                    supported_files.append(uploaded_file.name)
                    st.success(f"Processed TXT: {uploaded_file.name}")
                    
                else:
                    unsupported_files.append(uploaded_file.name)
                    st.warning(f"Unsupported format: {uploaded_file.name}")
                    
            except Exception as e:
                st.error(f"Error processing {uploaded_file.name}: {str(e)}")
                unsupported_files.append(uploaded_file.name)
            finally:
                # Clean up temporary file
                if os.path.exists(tmp_file_path):
                    os.unlink(tmp_file_path)
                    
            # Small delay to show progress
            time.sleep(0.1)
            
        except Exception as e:
            st.error(f"Error with {uploaded_file.name}: {str(e)}")
            unsupported_files.append(uploaded_file.name)
    
    # Clear progress indicators
    progress_bar.empty()
    status_text.empty()
    
    if not documents:
        return None
    
    # Show chunking process
    with st.spinner("Dividing documents into chunks..."):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        texts = text_splitter.split_documents(documents)
        st.success(f"Divided into {len(texts)} chunks")
    
    # Create vector store
    with st.spinner("Creating search index..."):
        try:
            vectorstore = FAISS.from_documents(texts, embeddings)
            st.success("Search index created")
            return vectorstore
        except Exception as e:
            st.error(f"Error creating vector store: {str(e)}")
            return None

# Process uploaded files
def setup_agent(uploaded_files):
    if uploaded_files:
        vectorstore = process_documents(uploaded_files)
        if vectorstore:
            st.session_state.vectorstore = vectorstore
            st.session_state.processed = True
            st.session_state.uploaded_files = [file.name for file in uploaded_files]
            return True
    return False

# Modern UI with default Streamlit layout
st.markdown('<h1 class="main-header">Multi-AI Document Assistant</h1>', unsafe_allow_html=True)
st.markdown('<p class="subheader">Upload documents and engage with multiple AI models</p>', unsafe_allow_html=True)

# Initial conversation starter
if not st.session_state.messages:
    st.session_state.messages.append({
        "role": "assistant", 
        "content": "Hello! I'm your document analysis assistant. Upload some documents and I'll help you explore their content. What would you like to discuss about your documents?",
        "timestamp": time.strftime("%H:%M:%S")
    })

# Show DOCX support status
if not DOCX_SUPPORT:
    st.warning("DOCX file support is not enabled. Please install: `pip install docx2txt`")

# Main chat interface
st.markdown("### Conversation")
chat_container = st.container()

with chat_container:
    # Create a scrollable chat area
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    
    for message in st.session_state.messages:
        if message["role"] == "user":
            st.markdown(f'''
                <div class="user-message">
                    <strong>You:</strong> {message["content"]}
                    <div class="message-time">{message.get("timestamp", "")}</div>
                </div>
            ''', unsafe_allow_html=True)
        else:
            formatted_content = format_response(message["content"])
            st.markdown(f'''
                <div class="assistant-message">
                    <strong>Assistant:</strong> {formatted_content}
                    <div class="message-time">{message.get("timestamp", "")}</div>
                </div>
            ''', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# Sidebar (default Streamlit layout)
with st.sidebar:
    st.header("AI Model Selection")
    
    # Model selector
    st.markdown('<div class="model-selector">', unsafe_allow_html=True)
    model_option = st.selectbox(
        "Choose AI Model",
        ["Google Gemini", "OpenAI GPT", "DeepSeek", "LLaMA"],
        index=0,
        help="Select which AI model to use for generating responses"
    )
    st.session_state.selected_model = model_option
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.header("Document Management")
    
    # File uploader with explicit acceptance
    uploaded_files = st.file_uploader(
        "Choose files", 
        type=["pdf", "csv", "txt"] + (["docx"] if DOCX_SUPPORT else []), 
        accept_multiple_files=True,
        help="Select PDF, CSV, TXT, or DOCX files to analyze"
    )
    
    # Store uploaded files in session state
    if uploaded_files:
        st.session_state.uploaded_files = [file.name for file in uploaded_files]
    
    # API key inputs based on selected model
    if st.session_state.selected_model == "Google Gemini":
        api_key = st.text_input("Google Gemini API Key", type="password", help="Get your API key from https://aistudio.google.com/")
    elif st.session_state.selected_model == "OpenAI GPT":
        api_key = st.text_input("OpenAI API Key", type="password", help="Get your API key from https://platform.openai.com/")
    elif st.session_state.selected_model == "DeepSeek":
        api_key = st.text_input("DeepSeek API Key", type="password", help="Get your API key from https://platform.deepseek.com/")
    elif st.session_state.selected_model == "LLaMA":
        api_key = st.text_input("Replicate API Key (for LLaMA)", type="password", help="Get your API key from https://replicate.com/")
    
    # Process button with loading state
    if st.button("Process Documents", type="primary", use_container_width=True):
        if uploaded_files:
            with st.spinner("Starting document processing..."):
                success = setup_agent(uploaded_files)
                if success:
                    st.success("Documents processed successfully!")
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": "Your documents have been processed successfully. You can now ask questions about their content.",
                        "timestamp": time.strftime("%H:%M:%S")
                    })
                else:
                    st.error("Failed to process documents")
        else:
            st.warning("Please upload documents first")
    
    # Show uploaded files
    if st.session_state.uploaded_files:
        st.divider()
        st.write("Uploaded Files:")
        for file_name in st.session_state.uploaded_files:
            st.markdown(f'<p class="file-name">• {file_name}</p>', unsafe_allow_html=True)
    
    st.divider()
    
    # Additional options
    st.write("Chat Options")
    streaming_enabled = st.checkbox("Enable streaming responses", value=True)
    
    if st.button("Clear Chat History", type="secondary", use_container_width=True):
        st.session_state.messages = []
        st.session_state.memory.clear()
        st.rerun()

# Chat input at the bottom
if prompt := st.chat_input("Ask a question about your documents..."):
    # Add user message to chat history
    st.session_state.messages.append({
        "role": "user", 
        "content": prompt,
        "timestamp": time.strftime("%H:%M:%S")
    })
    
    with st.spinner("Thinking..."):
        # Get relevant context from documents
        if st.session_state.vectorstore:
            docs = st.session_state.vectorstore.similarity_search(prompt, k=3)
            context = "\n".join([doc.page_content for doc in docs])
        else:
            context = "No documents processed yet."
        
        # Generate response using selected model
        if st.session_state.selected_model == "Google Gemini":
            response = query_gemini(prompt, context, api_key, stream=streaming_enabled)
        elif st.session_state.selected_model == "OpenAI GPT":
            response = query_openai(prompt, context, api_key, stream=streaming_enabled)
        elif st.session_state.selected_model == "DeepSeek":
            response = query_deepseek(prompt, context, api_key, stream=streaming_enabled)
        elif st.session_state.selected_model == "LLaMA":
            response = query_llama(prompt, context, api_key, stream=streaming_enabled)
        else:
            response = "Please select a valid AI model"
        
        # Add assistant response to chat history
        st.session_state.messages.append({
            "role": "assistant", 
            "content": response,
            "timestamp": time.strftime("%H:%M:%S")
        })
        
        st.rerun()