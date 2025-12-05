import streamlit as st
import os
import file_readers
from rag_pipeline import RAGPipeline

# st.markdown(
#     """
#     <style>
#     /* Hide list of uploaded files */
#     ul{
#         display: none
#     }
    
#     /* Hide "Showing page X of Y" in file uploader */
#     div[data-testid="stFileUploaderPagination"] {
#         display: none;
#     }
#     </style>
#     """,
#     unsafe_allow_html=True
# )


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
    st.toast("Loaded sample documents!", icon="📚")

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
# File Upload
# ----------------------------
uploaded_files = st.file_uploader(
    "Upload documents (.pdf, .docx, .txt, or .md)",
    type=["pdf", "docx", "txt", "md"],
    accept_multiple_files=True
)

# # Use these to detect when files are added or removed
# current_files = {f.name for f in uploaded_files} if uploaded_files else set()
# saved_files = {d["filename"] for d in st.session_state.docs}

# # Build documents + embeddings
# if uploaded_files is not None and current_files != saved_files:

#     # Reset docs list to clear out any files that were removed
#     st.session_state.docs = []

#     for f in uploaded_files:
#         # Get file extension
#         ext = f.name.split(".")[-1].lower()

#         if ext in ["txt", "md"]:
#             text = file_readers.read_txt(f)
#         elif ext == "pdf":
#             text = file_readers.read_pdf(f)
#         elif ext == "docx":
#             text = file_readers.read_docx(f)
#         else:
#             st.error(f"Unsupported type: {f.name}")
#             continue

#         st.session_state.docs.append({"filename": f.name, "text": text})

#     # Rebuild RAG pipeline with updated files
#     st.session_state.rag.add_documents(st.session_state.docs)

#     # UI notifcation
#     if len(current_files) > len(saved_files):
#         st.toast("Documents processed!", icon="📚")
#     else:
#         st.toast("Document removed!", icon="📚")


# ----------------------------
# Loaded Documents
# ----------------------------
st.write("### 📄 Loaded Documents")

for id, document in enumerate(st.session_state.docs):
    cols = st.columns([6,1])

    # Left column (Preview)
    with cols[0]:
        with st.expander(document["filename"], expanded = False):
            st.text_area(
                "Preview",
                document["text"],
                height = 300,
                disabled = True,
                key = f"preview_{id}"
            )
    
    # Right column (Trash)
    with cols[1]:
        delete_key = f"delete_{id}"
        if st.button("🗑️", key=delete_key):
            # Remove document from docs
            st.session_state.docs.pop(id)
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
st.write(
    "Now that you've supplied me with some documents to reference, " \
    "I can help answer any questions you may have about " \
    "how the game should work based on the documentation. " \
    "This way your team can spend more time focused on making the game " \
    "instead of searching for answers to common questions."
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