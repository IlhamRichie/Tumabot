from flask import Flask, request, jsonify
import torch
from langchain.chains import create_retrieval_chain
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceBgeEmbeddings

app = Flask(__name__)

# Fungsi untuk menginisialisasi LLM
def initialize_llm(groq_api_key):
    llm = ChatGroq(temperature=0, model_name="llama3-8b-8192", groq_api_key=groq_api_key)
    return llm

# Fungsi untuk menginisialisasi embeddings
def initialize_embeddings():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model_name = "BAAI/bge-small-en"
    model_kwargs = {"device": device}
    encode_kwargs = {"normalize_embeddings": True}
    embeddings = HuggingFaceBgeEmbeddings(
        model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs
    )
    return embeddings

# Buat chain retrieval-augmented generation
def create_rag_chain(retriever, llm):
    system_prompt = (
        "Anda adalah asisten untuk tugas menjawab pertanyaan yang bernama Tumabot. "
        "Gunakan konteks yang diambil untuk menjawab "
        "Menjawab menggunakan bahasa Indonesia "
        "Jika Anda tidak ada jawaban pada konteks, katakan saja saya tidak tahu dan berikan jawaban yang sesuai "
        ". Gunakan maksimal dua kalimat dan pertahankan jawaban singkat.\n\n"
        "{context}"
    )

    retrieval_qa_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{question}")
        ])
        | llm
        | StrOutputParser()
    )

    return retrieval_qa_chain

# Inisialisasi model
llm = initialize_llm(groq_api_key="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")
embeddings = initialize_embeddings()

# Load dataset dan buat retriever
pdf_loader = PyPDFLoader("dataset_tumanina.pdf")
documents = pdf_loader.load()
vectorstore = FAISS.from_documents(documents, embeddings)
retriever = vectorstore.as_retriever()

# Inisialisasi chain
rag_chain = create_rag_chain(retriever, llm)

# Endpoint Flask
@app.route('/tumabot', methods=['POST'])
def chat():
    data = request.json
    question = data.get('question', '')

    if not question:
        return jsonify({"error": "Question is required"}), 400

    try:
        response = rag_chain.invoke(question)
        return jsonify({"answer": response})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
