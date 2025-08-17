import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

# Set the Gemini API key via environment variable
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in environment variables.")

# Configure the API key
genai.configure(api_key=GOOGLE_API_KEY)

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=10000, chunk_overlap=1000
    )
    chunks = text_splitter.split_text(text)
    return chunks


def get_vector_store(text_chunks):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")


def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return prompt


def user_input(user_question):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)

    # Prepare context from retrieved documents
    context = "\n".join([doc.page_content for doc in docs])
    
    # Get the prompt template
    prompt_template = get_conversational_chain()
    
    # Format the prompt with context and question
    formatted_prompt = prompt_template.format(context=context, question=user_question)

    try:
        # pick one of the current models; 1.5-pro = best text quality, 1.5-flash = cheaper/faster
        model = genai.GenerativeModel("models/gemini-1.5-flash")
        resp = model.generate_content(formatted_prompt)     # or: model.generate_content([formatted_prompt])
        st.write("Reply: ", resp.text)
        print(resp.text)
  
    except AttributeError:
        # You’re still on an old SDK – tell the user cleanly
        st.error("Your google-generativeai package is too old. Run: pip install -U google-generativeai")
    except Exception as e:
        st.error(f"Error generating response: {e}")
        print(f"Error: {e}")



def main():
    st.set_page_config("Chat PDF")
    st.header("Chat with PDF using Gemini💁")

    user_question = st.text_input("Ask a Question from the PDF Files")

    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader(
            "Upload your PDF Files and Click on the Submit & Process Button",
            accept_multiple_files=True,
        )
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Done")


if __name__ == "__main__":
    main()