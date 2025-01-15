import os
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import DashScopeEmbeddings

load_dotenv()
embeddings = DashScopeEmbeddings(model="text-embedding-v2")
persist_directory = './db'

def load_pdfs_from_folder(folder_path):
    documents = []
    # 遍历指定文件夹下的所有文件和子文件夹
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith('.pdf'):  # 确保我们只处理PDF文件
                pdf_path = os.path.join(root, file)
                try:
                    loader = PyPDFLoader(pdf_path)
                    doc = loader.load()
                    documents.extend(doc)
                    print(f"Successfully loaded: {pdf_path}")
                except Exception as e:
                    print(f"Failed to load {pdf_path}: {e}")

    return documents

def load_documents_to_db():
    # 指定要加载PDF的文件夹路径
    folder_path = './docs'  # 替换为你的文件夹路径

    # 加载PDF文档
    pdf_documents = load_pdfs_from_folder(folder_path)

    # 打印加载的文档数量
    print(f"Loaded {len(pdf_documents)} PDF documents.")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        length_function=len,
        separators=['\n\n', '\n', ' ', '']
    )

    texts, metadatas = [], []
    for doc in pdf_documents:
        texts.append(doc.page_content)
        metadatas.append(doc.metadata)

    doc_splits = text_splitter.create_documents(texts, metadatas=metadatas)
    print(f"Total number of splits: {len(doc_splits)}")

    vectordb = Chroma.from_documents(
        documents = doc_splits,
        collection_name="rag-chroma",
        embedding = embeddings,
        persist_directory = persist_directory
    )
    return vectordb

# load_documents_to_db()

retriever = Chroma(
    collection_name="rag-chroma",
    persist_directory=persist_directory,
    embedding_function=DashScopeEmbeddings(model="text-embedding-v2"),
).as_retriever()
