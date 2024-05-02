import os
import pickle
import time
import streamlit as st
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate


# Configure Google Generative AI
from google.generativeai import configure

configure(api_key=st.secrets["GOOGLE_API_KEY"])

class DataLoader:
    def __init__(self, urls):
        self.urls = urls
        self.loader = UnstructuredURLLoader(urls=urls)

    def load_data(self):
        return self.loader.load()

class TextSplitter:
    def __init__(self, separators, chunk_size, chunk_overlap):
        self.separators = separators
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            separators=separators,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

    def split_documents(self, data):
        return self.text_splitter.split_documents(data)

class EmbeddingsGenerator:
    def __init__(self, model_path):
        self.embeddings = GoogleGenerativeAIEmbeddings(model=model_path)

    def generate_embeddings(self, documents):
        vectorstore = FAISS.from_documents(documents=documents, embedding=self.embeddings)
        return vectorstore

class StreamlitApp:
    def __init__(self, file_path):
        self.file_path = file_path
        self.query = None
        self.vectorIndex = None

    def load_data(self, urls):
        data_loader = DataLoader(urls)
        return data_loader.load_data()

    def split_documents(self, data):
        text_splitter = TextSplitter(separators=["\n\n", "\n", " "], chunk_size=1000, chunk_overlap=100)
        return text_splitter.split_documents(data)

    def generate_embeddings(self, docs):
        embeddings_generator = EmbeddingsGenerator(model_path="models/embedding-001")
        return embeddings_generator.generate_embeddings(docs)

    def run(self):
        st.title("InfoGenie")
        st.subheader("Type your query and let's uncover some answers! 🕵️‍♂️")

        st.sidebar.title("Add URLs on topics you'd like to explore and see the magic unfold! ✨")
        urls = [st.sidebar.text_input(f"URL{i+1}") for i in range(4)]
        process = st.sidebar.button("Run 🚀")

        if process:
            data = self.load_data(urls)
            docs = self.split_documents(data)
            vectorstore = self.generate_embeddings(docs)

            with open(self.file_path, "wb") as f:
                pickle.dump(vectorstore, f)

        self.query = st.text_input(" ")
        if self.query:
            if os.path.exists(self.file_path):
                with open(self.file_path, "rb") as f:
                    self.vectorIndex = pickle.load(f)
                    self.process_query()

    def process_query(self):
        llm_prompt_template = """You are an assistant for question-answering tasks.
        Use the following context to answer the question.
        If you don't know the answer, just say that you don't know.
        Use five sentences maximum and keep the answer concise.\n
        Question: {question} \nContext: {context} \nAnswer:"""
        retriever = self.vectorIndex.as_retriever(search_kwargs={"k": 6})

        llm_prompt = PromptTemplate.from_template(llm_prompt_template)

        # Chain for processing query
        rag_chain = (
            {"context": retriever | (lambda docs: "\n\n".join(doc.page_content for doc in docs)),
             "question": RunnablePassthrough()}
            | llm_prompt
            | ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.7, top_p=0.85)
            | StrOutputParser()
        )
        result = rag_chain.invoke(self.query)

        # Display result
        st.subheader("Answer 💡")
        st.write(result)

if __name__ == "__main__":
    app = StreamlitApp(file_path=r"model/faiss_store_gemini.pkl")
    app.run()