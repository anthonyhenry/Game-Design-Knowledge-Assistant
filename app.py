import streamlit as st
import file_readers
from rag_pipeline import RAGPipeline

st.set_page_config(
    page_title="Game Design Knowledge Assistant",
    page_icon="🎮"
)

st.title("🎮 Game Design Knowledge Assistant")

# Load pipeline once
@st.cache_resource
def load_rag_pipeline():
    return RAGPipeline()

if "rag" not in st.session_state:
    st.session_state.rag = load_rag_pipeline()

if "docs" not in st.session_state:
    st.session_state.docs = []

# ----------------------------
# File Upload
# ----------------------------
uploaded_files = st.file_uploader(
    "Upload documents (.pdf, .docx, .txt, or .md)",
    type=["pdf", "docx", "txt", "md"],
    accept_multiple_files=True
)

if uploaded_files:
    for f in uploaded_files:
        # Get file extension
        ext = f.name.split(".")[-1].lower()

        if ext in ["txt", "md"]:
            text = file_readers.read_txt(f)
        elif ext == "pdf":
            text = file_readers.read_pdf(f)
        elif ext == "docx":
            text = file_readers.read_docx(f)
        else:
            st.error(f"Unsupported type: {f.name}")
            continue

        st.session_state.docs.append({"filename": f.name, "text": text})

    st.success("Files uploaded! Now click **Process Documents** below.")

# ----------------------------
# Process Documents Button
# ----------------------------
if st.button("📚 Process Documents"):
    if not st.session_state.docs:
        st.error("Upload documents first.")
    else:
        rag = st.session_state.rag
        rag.add_documents(st.session_state.docs)
        st.success("Documents processed and embedded!")

# ----------------------------
# Query
# ----------------------------
question = st.text_input("Ask a question:")
if st.button("Submit Question"):
    if not st.session_state.docs:
        st.error("Upload and process documents first.")
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