import streamlit as st
import os
import file_readers
from rag_pipeline import RAGPipeline
import random
from datetime import datetime
import time
from llm_client import get_groq_client, get_llm_response

st.set_page_config(
    page_title="Ludexra",
    page_icon="🎮"
)

# ----------------------------
# Helper functions
# ----------------------------

def read_document(file, filename):
    ext = filename.split(".")[-1].lower()

    if ext in ["txt", "md"]:
        return file_readers.read_txt(file)
    elif ext == "pdf":
        return file_readers.read_pdf(file)
    elif ext == "docx":
        return file_readers.read_docx(file)
    
    return None

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# @st.cache_data # Cache sample docs so they don't load each rerun for performance
def load_sample_docs():
    SAMPLE_DOCS_DIR = os.path.join(ROOT_DIR, "sample_docs")

    sample_docs = []

    for filename in os.listdir(SAMPLE_DOCS_DIR):
        path = os.path.join(SAMPLE_DOCS_DIR, filename)

        # Get sample doc contents
        with open(path, "rb") as f:
            text = read_document(f, filename)
        if text is None:
            continue

        sample_docs.append({"filename": filename, "text": text})
    return sample_docs

# Set unique filenames for handling duplicate files
def get_unique_filename(filename, existing_filenames):
    if filename not in existing_filenames:
        return filename
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return f"{filename} [{timestamp}]"

# Split Documents into pages
PAGE_SIZE = 500 # Characters per page
def get_document_pages(text):
    # Split document into paragraphs
    paragraphs = text.split("\n")

    # Loop through paragraphs to create pages
    pages = []
    current_page = ""
    for paragraph in paragraphs:
        # Start a new page if adding this paragraph would exceed PAGE_SIZE
        if len(current_page) + len(paragraph) > PAGE_SIZE and len(current_page) > 0:
            pages.append(current_page.rstrip())
            current_page = ""
        current_page += paragraph + "\n"

    if current_page:
        pages.append(current_page.rstrip())

    return pages

def get_response_imgs():
    IMGS_DIR = os.path.join(ROOT_DIR, "imgs")

    imgs = []

    for filename in sorted(os.listdir(IMGS_DIR)):
        if "answer" in filename:
            imgs.append(os.path.join(IMGS_DIR, filename))
    
    return imgs
response_imgs = get_response_imgs()

# ----------------------------
# Initialize session state variables
# ----------------------------

# Cache rag pipeline (for performance)
@st.cache_resource
def load_rag_pipeline():
    return RAGPipeline()
if "rag" not in st.session_state:
    st.session_state.rag = load_rag_pipeline()

if "pending_toast" in st.session_state:
    # Display pending toast messages
    st.toast(st.session_state.pending_toast, icon="📚")
    del st.session_state.pending_toast

if "docs" not in st.session_state:
    st.session_state.docs = load_sample_docs()
    st.session_state.rag.add_documents(st.session_state.docs)

# Use a key for file uploader to clear files once processed
if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0

# Save conversation history
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []

# Show example questions based on sample docs
if "show_examples" not in st.session_state:
    st.session_state.show_examples = True
sample_questions = [
    "Can you summarize the core gameplay loop please?",
    "How does difficulty increase over time?",
    "What are the different alien types and how do they behave?",
    "How is player score calculated?",
    "What happens when all aliens are destroyed?",
    "What happens when the player loses all their lives?",
    "How does the player lose lives?",
    "How do the defensive bunkers work?"
]

# ----------------------------
# Use CSS to hide uploaded files
# ----------------------------
st.markdown(
    """
    <style>
    /* Hide list of uploaded files */
    ul{
        display: none
    }
    
    /* Hide "Showing page X of Y" in file uploader */
    div[data-testid="stFileUploaderPagination"] {
        display: none;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ----------------------------
# Welcome
# ----------------------------

greet_cols = st.columns([4,6])
with greet_cols[0]:
    st.image("imgs/ludexra-greet.png")
st.write(
    "Hello, I'm __Ludexra__!" \
)
st.write(
    "It takes a lot of documentation to make a video game. " \
    "I'm here to help you keep track of all the details related to the game you're working on! " \
    "Upload your documents to turn me into an expert on your game. "
    "I'll be able to answer any questions you have about the project, well, as long as the answer exists in the documentation." \
)
st.write(
    "Some sample documents are provided to get you started. " \
    "You can preview them in the Loaded Documents section.  "
)

# ----------------------------
# Upload Documents
# ----------------------------

# File uploader
uploaded_files = st.file_uploader(
    "Upload documents (.pdf, .docx, .txt, or .md)",
    type=["pdf", "docx", "txt", "md"],
    accept_multiple_files=True,
    key=f"uploader_{st.session_state.uploader_key}"
)

# Process new uploads
if uploaded_files:
    new_docs = []

    # Build a set of existing filenames
    existing_filenames = {
        d["filename"]
        for d in st.session_state.docs
    }

    for file in uploaded_files:
        # Get file contents
        text = read_document(file, file.name)
        if text is None:
            st.error(f"Unsupported type: {file.name}")
            continue

        # Use a unique name for the file
        doc_filename = get_unique_filename(file.name, existing_filenames)
        # Add to docs list
        doc = {"filename": doc_filename, "text": text}
        st.session_state.docs.append(doc)
        # Update existing filenames
        existing_filenames.add(doc_filename)

        new_docs.append(doc)


    # Update RAG pipeline only if new docs were added
    if new_docs:
        st.session_state.rag.add_documents(st.session_state.docs)
        st.session_state.pending_toast = (f"Processed {len(new_docs)} new document(s)!")
        st.session_state.uploader_key += 1
        st.rerun()

# ----------------------------
# Load Documents
# ----------------------------

st.write("### 📄 Loaded Documents")

for document in st.session_state.docs:
    # Load document pages
    pages = get_document_pages(document["text"])

    # Always save the last opened page of a document
    page_key = f"page_{document['filename']}"
    if page_key not in st.session_state:
        st.session_state[page_key] = 0
    current_page = st.session_state[page_key]
    
    # Create two columns
    preview_col, trash_col = st.columns([6, 1])
    
    # Preview Column
    with preview_col:
        # Display page preivew
        with st.expander(document["filename"], expanded=False):
            st.code(
                pages[current_page],
                language=None,
                wrap_lines=True
            )

            # Page navigation
            st.caption(f"Page {current_page + 1} of {len(pages)}")
            if len(pages) > 1:
                new_page = st.slider(
                    "Page Slider",
                    min_value=1,
                    max_value=len(pages),
                    value=current_page + 1,
                    key=f"slider_{document['filename']}",
                    # label_visibility="collapsed"
                )
                # Page changing
                if new_page - 1 != current_page:
                    st.session_state[page_key] = new_page - 1
                    st.rerun()
    # Trash column
    with trash_col:
        delete_key = f"delete_{document['filename']}"
        if st.button("🗑️", key=delete_key):
            # Stop saving the last opened page for the document
            page_key = f"page_{document['filename']}"
            if page_key in st.session_state:
                del st.session_state[page_key]

            # Remove doc
            st.session_state.docs = [
                d for d in st.session_state.docs if d["filename"] != document["filename"]
            ]

            # Hide example questions
            st.session_state.show_examples = False
            
            # Update RAG pipeline
            st.session_state.rag.add_documents(st.session_state.docs)
            
            # Rerun to update loaded documents list properly
            st.rerun()

# ----------------------------
# Query + Response
# ----------------------------

# st.write("### ❓ Query")
# st.write(
#     "I can help answer any questions you may have about " \
#     "how the game should work based on the documentation. " \
#     "I can also help you come up with new ideas for the game."
# )

with st.form("question_form", clear_on_submit=True):
    question = st.text_input("Ask a question:")
    submitted = st.form_submit_button("Submit Question")
if submitted:
    if not st.session_state.docs:
        st.error("Upload documents first.")
    elif not question.strip():
        st.error("Enter a question.")
    else:
        cols = st.columns([3, 8])
        with cols[0]:
            img_placeholder = st.empty()
            img_placeholder.image("imgs/ludexra-think.png")

        # Initialize Groq client
        if "groq_client" not in st.session_state:
            st.session_state.groq_client = get_groq_client()

        with st.spinner("Reviewing documents..."):
            # Build context using RAG
            context, sources = st.session_state.rag.build_context(question)

            # # Debug: Show top_k chunks for testing
            # st.subheader("📌 Retrieved Context")
            # for s in sources:
            #     st.write(f"**From {s['source']}** (score={s['score']:.3f})")
            #     st.code(s["chunk"][:400] + "...")

        with st.spinner("Forming response..."):
            # Call Groq LLM
            llm_answer = get_llm_response(st.session_state.groq_client, question, context)

        # Save question and response in conversation history
        st.session_state.conversation_history.append({
            "img": response_imgs[len(st.session_state.conversation_history) % len(response_imgs)],
            "question": question,
            "answer": llm_answer
        })

        # Hide examples and thinking face
        time.sleep(1) # 1 sec delay so thinking face removal isn't too jarring
        st.session_state.show_examples = False
        img_placeholder.empty()

if st.session_state.show_examples:
    st.caption("Try asking:")
    st.caption(random.choice(sample_questions))

# Display conversation history
for exchange in reversed(st.session_state.conversation_history):    
    st.write(exchange["question"])

    avatar_col, response_col = st.columns([2, 9])
    with avatar_col:
        st.image(exchange["img"])
    with response_col:
        st.write(exchange["answer"])

    st.divider()