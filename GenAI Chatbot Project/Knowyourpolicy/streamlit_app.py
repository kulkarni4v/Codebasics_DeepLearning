"""
Streamlit UI for Health Insurance Policy /Agreements RAG Chatbot
Beautiful, modern interface with streaming responses like ChatGPT

Installation:
pip install streamlit streamlit-chat

Run:
streamlit run streamlit_app.py
"""

import streamlit as st
import time
from main import load_and_process_pdf, create_vector_store, load_vector_store, create_rag_chain
import os

# ============================================================================
# PAGE CONFIGURATION - Must be first Streamlit command
# ============================================================================
st.set_page_config(
    page_title="Know Your Policy (KYP)",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CUSTOM CSS FOR MODERN LOOK
# ============================================================================
st.markdown("""
<style>
    /* Main container styling */
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Chat message containers */
    .stChatMessage {
        border-radius: 15px;
        padding: 15px;
        margin: 10px 0;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.3);
    }
    
    /* User message styling - Dark purple/blue background with white text */
    .stChatMessage[data-testid="user-message"] {
        background: linear-gradient(135deg, #1e3a8a 0%, #312e81 100%);
    }
    
    .stChatMessage[data-testid="user-message"] p,
    .stChatMessage[data-testid="user-message"] div {
        color: white !important;
    }
    
    /* Assistant message styling - Dark gray background with white text */
    .stChatMessage[data-testid="assistant-message"] {
        background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
        border-left: 4px solid #667eea;
    }
    
    .stChatMessage[data-testid="assistant-message"] p,
    .stChatMessage[data-testid="assistant-message"] div {
        color: white !important;
    }
    
    /* Input box styling */
    .stChatInputContainer {
        border-top: 2px solid rgba(255, 255, 255, 0.1);
        padding-top: 20px;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #2c3e50 0%, #34495e 100%);
    }
    
    [data-testid="stSidebar"] * {
        color: white !important;
    }
    
    /* Title styling */
    h1 {
        color: white;
        text-align: center;
        padding: 20px 0;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3);
    }
    
    /* Success message styling */
    .element-container div[data-testid="stMarkdownContainer"] p {
        color: white;
    }
    
    /* Spinner styling */
    .stSpinner > div {
        border-top-color: white !important;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 10px 24px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0, 0, 0, 0.3);
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# INITIALIZE SESSION STATE
# ============================================================================
# Session state keeps data persistent across reruns
if "messages" not in st.session_state:
    # messages: Stores the full chat history for display
    st.session_state.messages = []

if "chain" not in st.session_state:
    # chain: Stores the RAG chain (initialized once)
    st.session_state.chain = None

if "vectorstore_loaded" not in st.session_state:
    # vectorstore_loaded: Tracks if the database is ready
    st.session_state.vectorstore_loaded = False

if "current_pdf" not in st.session_state:
    # current_pdf: Stores the name of the currently loaded PDF
    st.session_state.current_pdf = None

# ============================================================================
# SIDEBAR - CONFIGURATION AND INFO
# ============================================================================
with st.sidebar:
    st.markdown("### 🏥 Policy Assistant")
    st.markdown("---")

    # PDF upload/path selection
    st.markdown("#### 📄 Document Settings")

    # Choose between upload or file path
    upload_option = st.radio(
        "Select document source:",
        ["📤 Upload PDF", "📁 Use File Path"],
        help="Upload a new PDF or use an existing file path"
    )

    pdf_path = None
    uploaded_file = None

    if upload_option == "📤 Upload PDF":
        # File uploader widget
        uploaded_file = st.file_uploader(
            "Upload your insurance policy/Any document PDF",
            type=['pdf'],
            help="Maximum file size: 200MB"
        )

        if uploaded_file is not None:
            # Save uploaded file temporarily
            temp_pdf_path = f"temp_uploads/{uploaded_file.name}"
            os.makedirs("temp_uploads", exist_ok=True)

            with open(temp_pdf_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            pdf_path = temp_pdf_path
            st.success(f"✓ Uploaded: {uploaded_file.name}")

    else:  # Use File Path
        pdf_path = st.text_input(
            "PDF File Path",
            value="data_pdf_file/care-supreme---policy-terms-&-conditions-(effective-from-19-march-2025).pdf",
            help="Path to your insurance policy PDF on disk"
        )

    # Database path configuration
    db_path = st.text_input(
        "Database Path",
        value="./chroma_db",
        help="Local path to store/load the vector database"
    )

    st.markdown("---")

    # Initialize/Load button
    initialize_disabled = pdf_path is None or pdf_path == ""

    if st.button("🔄 Initialize Chatbot", use_container_width=True, disabled=initialize_disabled):
        with st.spinner("Loading your policy..."):
            try:
                # Validate PDF exists
                if not os.path.exists(pdf_path):
                    st.error(f"❌ PDF file not found: {pdf_path}")
                    st.stop()

                # Check if database exists
                if not os.path.exists(db_path):
                    st.info("📄 Processing PDF for the first time...")

                    # Load and process PDF
                    chunks = load_and_process_pdf(pdf_path)

                    # Create vector store
                    vectorstore = create_vector_store(chunks, db_path)

                    st.success("✓ Policy processed and database created!")
                else:
                    # Ask if user wants to reprocess or use existing
                    st.warning("⚠️ Database already exists!")
                    reprocess = st.checkbox("Reprocess PDF (will delete existing database)")

                    if reprocess:
                        import shutil
                        shutil.rmtree(db_path)
                        st.info("📄 Reprocessing PDF...")

                        chunks = load_and_process_pdf(pdf_path)
                        vectorstore = create_vector_store(chunks, db_path)

                        st.success("✓ Policy reprocessed!")
                    else:
                        st.info("📂 Loading existing database...")
                        vectorstore = load_vector_store(db_path)
                        st.success("✓ Database loaded!")

                # Create RAG chain
                st.session_state.chain = create_rag_chain(vectorstore)
                st.session_state.vectorstore_loaded = True

                # Store current PDF name
                st.session_state.current_pdf = os.path.basename(pdf_path)

                st.success("✓ Chatbot ready!")

            except Exception as e:
                st.error(f"❌ Error: {str(e)}")

    st.markdown("---")

    # Clear conversation button
    if st.button("🗑️ Clear Conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.markdown("---")

    # Information section
    st.markdown("#### ℹ️ About")

    # Show currently loaded PDF
    if st.session_state.current_pdf:
        st.info(f"📄 Current: {st.session_state.current_pdf}")

    st.markdown("""
    This AI assistant helps you understand your personal documents like
    Insurance Policies/Agreements/Any other document .
    
    **Features:**
    - 📖 Strictly Document-based answers only
    - 💬 Conversation memory
    - 🎯 No hallucinations
    - ⚡ Instant responses
    - 📤 Upload any PDF policy/agreement/any document
    
    **Tips:**
    - Ask specific questions
    - Reference previous answers
    - Request clarifications - IMPORTANT
    """)

    st.markdown("---")
    st.markdown("**Model:** GPT-4.1-mini")
    st.markdown("**Embeddings:** text-embedding-3-large")

# ============================================================================
# MAIN CHAT INTERFACE
# ============================================================================

# Title
st.markdown("<h1>🏥 Health Insurance Policy Assistant</h1>", unsafe_allow_html=True)

# Check if chatbot is initialized
if not st.session_state.vectorstore_loaded:
    st.info("👈 Please initialize the chatbot using the sidebar button")
    st.stop()

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ============================================================================
# STREAMING RESPONSE FUNCTION
# ============================================================================
def stream_response(response_text):
    """
    Stream response character by character
    Preserves spaces, newlines, markdown formatting
    """
    for char in response_text:
        yield char
        if char in ".!?":
            time.sleep(0.3)
        else:
            time.sleep(0.08)
# ============================================================================
# CHAT INPUT AND RESPONSE
# ============================================================================

# Chat input field (fixed at bottom)
if prompt := st.chat_input("Ask about your insurance policy..."):

    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate assistant response
    with st.chat_message("assistant"):
        # Placeholder for streaming response
        message_placeholder = st.empty()
        full_response = ""

        try:
            # Get response from RAG chain
            with st.spinner("Thinking..."):
                result = st.session_state.chain({"question": prompt})
                assistant_response = result["answer"]

            # Stream the response word by word
            for chunk in stream_response(assistant_response):
                full_response += chunk
                message_placeholder.markdown(full_response + "▌", unsafe_allow_html=False)

            message_placeholder.markdown(full_response)

            # Final response without cursor
            message_placeholder.markdown(full_response)

            # Optional: Show source documents in an expander
            # with st.expander("📄 View Sources"):
            #     sources = result.get("source_documents", [])
            #     if sources:
            #         for i, doc in enumerate(sources, 1):
            #             st.markdown(f"**Source {i}** (Page {doc.metadata.get('page', 'N/A')})")
            #             st.text(doc.page_content[:300] + "...")
            #             st.markdown("---")
            #     else:
            #         st.info("No specific sources referenced")

        except Exception as e:
            full_response = f"❌ An error occurred: {str(e)}"
            message_placeholder.markdown(full_response)

    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": full_response})

# ============================================================================
# FOOTER
# ============================================================================
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: white; opacity: 0.7;'>"
    "🤖 Powered by LangChain, OpenAI, and ChromaDB | "
    "💡 Answers based on policy documents only"
    "</p>",
    unsafe_allow_html=True
)