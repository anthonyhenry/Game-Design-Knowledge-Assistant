import streamlit as st
import file_readers


st.set_page_config(
    page_title="Game Design Knowledge Assistant",
    page_icon="🎮",
    # layout="wide"
)

st.title("🎮 Game Design Knowledge Assistant")

if "docs" not in st.session_state:
    st.session_state.docs = []

# st.write("Upload your game design documents!")

# Document upload
uploaded_files = st.file_uploader(
    "Upload documents (.pdf, .docx, .txt or .md )",
    type=["txt", "md", "pdf", "docx"],
    accept_multiple_files=True
)

if uploaded_files:
    # st.success(f"{len(uploaded_files)} document(s) uploaded!")
    for f in uploaded_files:
        # st.write(f"📄 {f.name}")

        # Get file extension
        ext = f.name.split(".")[-1].lower()

        if ext in ["txt", "md"]:
            text = file_readers.read_txt(f)
        elif ext == "pdf":
            text = file_readers.read_pdf(f)
        elif ext == "docx":
            text = file_readers.read_docx(f)
        else:
            st.error(f"Unsupported file type: {f.name}")
            continue

        st.session_state.docs.append({"filename": f.name, "text": text})

st.write("### Loaded Documents:")
for d in st.session_state.docs:
    st.write(f"- {d['filename']}")

if st.checkbox("Show extracted text"):
    for d in st.session_state.docs:
        st.subheader(d["filename"])
        st.code(d["text"][:1000] + "...")

# User question
question = st.text_input("Ask a question about your design documents:")

if st.button("Submit Query"):
    if not uploaded_files:
        st.error("Please upload at least one document first.")
    elif not question.strip():
        st.error("Please enter a question.")
    else:
        st.info("The RAG pipeline will answer here once implemented!")