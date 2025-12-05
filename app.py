import streamlit as st
import os
import file_readers
from rag_pipeline import RAGPipeline


# ----------------------------
# CSS to hide uploaded files
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
# Cache rag pipeline (for performance)
# ----------------------------
@st.cache_resource
def load_rag_pipeline():
    return RAGPipeline()

if "rag" not in st.session_state:
    st.session_state.rag = load_rag_pipeline()

# ----------------------------
# Load sample documents on first run
# ----------------------------
def load_sample_docs():
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    SAMPLE_DOCS_DIR = os.path.join(ROOT_DIR, "sample_docs")

    sample_docs = []

    for filename in os.listdir(SAMPLE_DOCS_DIR):
        path = os.path.join(SAMPLE_DOCS_DIR, filename)
        ext = filename.split(".")[-1].lower()

        with open(path, "rb") as f:
            if ext in ["txt", "md"]:
                text = file_readers.read_txt(f)
            elif ext == "pdf":
                text = file_readers.read_pdf(f)
            elif ext == "docx":
                text = file_readers.read_docx(f)
            else:
                continue

        sample_docs.append({"filename": filename, "text": text})
    return sample_docs

if "docs" not in st.session_state:
    st.session_state.docs = load_sample_docs()
    st.session_state.rag.add_documents(st.session_state.docs)

# ----------------------------
# Begin UI
# ----------------------------
st.set_page_config(
    page_title="Game Design Knowledge Assistant",
    page_icon="🎮"
)

st.title("🎮 Game Design Knowledge Assistant 📓")
st.write(
    "Hello, I'm the Game Design Knowledge Assistant! " \
    "It takes a lot of documentation to make a video game. " \
    "I can help you and your team keep track of all the documents related to your game. " \
    "Some sample documents are provided to get you started. " \
    "You can preview them in the Loaded Documents section,  " \
    "then ask me questions about them. " \
    "When you're ready, you can upload your own documents for me to reference."
)


# ----------------------------
# Load Documents
# ----------------------------

# Track processed files since uploaded files persist
if "processed_files" not in st.session_state:
    st.session_state.processed_files = set()

# File uploader
uploaded_files = st.file_uploader(
    "Upload documents (.pdf, .docx, .txt, or .md)",
    type=["pdf", "docx", "txt", "md"],
    accept_multiple_files=True,
)

# Process new uploads
if uploaded_files:
    new_docs = []

    for file in uploaded_files:
        if file.name in st.session_state.processed_files:
            continue  # skip files already processed

        # Remove old doc with same filename (overwrite behavior)
        st.session_state.docs = [
            d for d in st.session_state.docs if d["filename"] != file.name
        ]

        # Read file contents
        ext = file.name.split(".")[-1].lower()
        if ext in ["txt", "md"]:
            text = file_readers.read_txt(file)
        elif ext == "pdf":
            text = file_readers.read_pdf(file)
        elif ext == "docx":
            text = file_readers.read_docx(file)
        else:
            st.error(f"Unsupported type: {file.name}")
            continue

        # Add to docs list
        doc = {"filename": file.name, "text": text}
        st.session_state.docs.append(doc)
        new_docs.append(doc)

        # Mark as processed
        st.session_state.processed_files.add(file.name)

    # Update RAG pipeline only if new docs were added
    if new_docs:
        st.session_state.rag.add_documents(st.session_state.docs)
        st.toast(f"Processed {len(new_docs)} new document(s)!", icon="📚")

# ----------------------------
# Loaded Documents
# ----------------------------
st.write("### 📄 Loaded Documents")

for document in st.session_state.docs:
    # Create two columns
    cols = st.columns([6, 1])
    
    # Preview Column
    with cols[0]:
        with st.expander(document["filename"], expanded=False):
            st.text_area(
                "Preview",
                document["text"],
                height=300,
                disabled=True,
                key=f"preview_{document['filename']}"
            )
    # Trash column
    with cols[1]:
        delete_key = f"delete_{document['filename']}"
        if st.button("🗑️", key=delete_key):
            # Remove doc
            st.session_state.docs = [
                d for d in st.session_state.docs if d["filename"] != document["filename"]
            ]
            # Update RAG pipeline
            st.session_state.rag.add_documents(st.session_state.docs)
            # Rerun to update loaded documents list properly
            st.rerun()

# ----------------------------
# Query
# ----------------------------

# print(len(uploaded_files))

# Only show query section if there are documents to refernce
# if(len(uploaded_files) > 0):
st.write("### ❓ Query")
st.write(
    "I can help answer any questions you may have about " \
    "how the game should work based on the documentation. " \
    "I can also help you come up with new ideas for the game."
)

question = st.text_input("Ask a question:")
if st.button("Submit Question"):
    if not st.session_state.docs:
        st.error("Upload documents first.")
    elif not question.strip():
        st.error("Enter a question.")
    else:
        rag = st.session_state.rag

        from llm_client import get_groq_client, get_llm_response

        # Initialize Groq client
        if "groq_client" not in st.session_state:
            st.session_state.groq_client = get_groq_client()

        # Build context using RAG
        context, sources = st.session_state.rag.build_context(question)

        st.subheader("📌 Retrieved Context")
        for s in sources:
            st.write(f"**From {s['source']}** (score={s['score']:.3f})")
            st.code(s["chunk"][:400] + "...")

        # Call Groq LLM
        llm_answer = get_llm_response(st.session_state.groq_client, question, context)

        st.subheader("💬 Assistant Response")
        st.write(llm_answer)