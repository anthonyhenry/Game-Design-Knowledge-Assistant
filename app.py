import streamlit as st

st.title("🎮 Game Design Knowledge Assistant")

st.write("Upload your game design documents!")

# Document upload
uploaded_files = st.file_uploader(
    "Upload .txt or .md documents",
    type=["txt", "md"],
    accept_multiple_files=True
)

if uploaded_files:
    st.success(f"{len(uploaded_files)} document(s) uploaded!")
    for f in uploaded_files:
        st.write(f"📄 {f.name}")

# User question
question = st.text_input("Ask a question about your design documents:")

if st.button("Submit Query"):
    if not uploaded_files:
        st.error("Please upload at least one document first.")
    elif not question.strip():
        st.error("Please enter a question.")
    else:
        st.info("The RAG pipeline will answer here once implemented!")