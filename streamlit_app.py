import streamlit as st
import os
from dotenv import load_dotenv
import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from streamlit_lottie import st_lottie
import requests
import io

# Load environment variables
load_dotenv()

GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')

# Function to load Lottie animations
def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# Load Lottie animations
lottie_document = load_lottieurl("https://lottie.host/166bd88a-d9d0-498b-8f86-774195b99454/MBWDTWyffi.json")
lottie_chat = load_lottieurl("https://lottie.host/df41fd21-d0a2-4904-8e86-6d266299bca2/m6FgBZPrax.json")

# PDF processing functions
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        try:
            pdf_reader = PdfReader(io.BytesIO(pdf.read()))
            for page in pdf_reader.pages:
                text += page.extract_text() or ""
        except Exception as e:
            st.error(f"Error processing {pdf.name}: {str(e)}")
            st.error("This file may not be a valid PDF or might be corrupted. Please check the file and try again.")
            return None
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

template = """
You are an AI legal advisor chatbot. Your task is to answer questions based solely on the content of the uploaded PDF document. 

Given the following extracted parts of a legal document and a question, create a final answer.

context: {context}

question: {question}

If the question is not related to the content of the PDF, politely decline to answer and explain that you can only provide information based on the uploaded document.

Answer:
"""

def get_conversational_chain():
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(input_variables=["question", "context"], template=template)
    chains = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chains

def load_faiss_index(pickle_file):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    faiss_index = FAISS.load_local(pickle_file, embeddings=embeddings, allow_dangerous_deserialization=True)
    return faiss_index

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = load_faiss_index("faiss_index")
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question})

    st.write("Answer:", response["output_text"])

def generate_summary(text):
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.5)
    prompt = "Summarize the following legal document in a concise manner:\n\n" + text
    response = model.invoke(prompt)
    return response.content

# Page config
st.set_page_config(page_title='DocWhisperer', layout='wide', page_icon="🗂️")

# Custom CSS
st.markdown("""
<style>
    body {
        background-color: #f0f2f6;
        color: #333333;
        font-family: 'Arial', sans-serif;
    }
    .main {
        background-color: #ffffff;
        border-radius: 15px;
        padding: 30px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
        border-radius: 20px;
        padding: 10px 20px;
        border: none;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #45a049;
        transform: translateY(-2px);
    }
    .css-1cpxqw2 {
        background-color: #ffffff;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .stTextInput>div>div>input {
        border-radius: 20px;
    }
    .stAlert {
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image('logo/Sahi Jawab.png', use_column_width=True, caption='Sahi Jawab : Your Nyaya Mitra 👩🏻‍⚖️📚𓍝')
    st.markdown("---")
    st.markdown("### About DocWhisperer")
    st.info("DocWhisperer allows you to upload your legal documents and chat with their content. Get insights and answers specific to your documents!")

# Main content
st.title("🗂️ DocWhisperer: Chat with Your Legal Documents")

# Display document animation before file upload
if lottie_document:
    st_lottie(lottie_document, height=300, key="document")

# File uploader
uploaded_files = st.file_uploader("Upload your legal document (PDF)", type="pdf", accept_multiple_files=True)

if uploaded_files:
    with st.spinner("Processing your document..."):
        # Read PDF and get text
        raw_text = get_pdf_text(uploaded_files)
        
        if raw_text is not None:
            # Generate and display summary
            summary = generate_summary(raw_text)
            st.subheader("PDF Summary")
            st.write(summary)

            # Create text chunks and vector store
            text_chunks = get_text_chunks(raw_text)
            vector_store = get_vector_store(text_chunks)

            st.success("Document processed successfully! You can now ask questions about its content.")

            # Chat interface
            if lottie_chat:
                st_lottie(lottie_chat, height=200, key="chat")

            user_question = st.text_input("Ask a question about your document:")
            if user_question:
                user_input(user_question)
        else:
            st.error("Unable to process the document. Please upload a valid PDF file.")

# Display warning
st.warning("Please note that DocWhisperer provides information based on the uploaded document. Always consult with a qualified legal professional for accurate legal advice.")

# Footer
st.markdown("""
---
<p style="text-align: center; color: #666666;">© 2024 Sahi Jawab - AI Legal Advisor. All rights reserved.</p>
""", unsafe_allow_html=True)
