from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredMarkdownLoader
import os
import time

ollama_url = os.environ.get("OLLAMA_API_BASE")


def md_rag(
    source_dir,
    db_path,
    collection_name,
    model_name: str = "mxbai-embed-large:latest",
    chunk_size=1000,
    chunk_overlap=150,
):
    """
    Creates or updates a vector database from MD files in a specified folder.
    Only adds documents that are not already present in the database.

    Args:
        source_directory (str): The path to the folder containing MD files.
        db_path (str): The directory to store the Chroma vector database.
        model_name (str): The name of the Ollama embedding model to use.
        chunk_size (int): The size of each text chunk when splitting documents.
        chunk_overlap (int): The number of overlapping characters between chunks.
    """
    start_time = time.time()
    add_documents = not os.path.exists(db_path)
    embeddingFn = OllamaEmbeddings(model=model_name)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size)

    if add_documents:
        print(f"DB is not created, creating db @ {db_path}")
        documents = []
        ids = []
        for fileName in os.listdir(source_dir):
            if fileName.endswith(".md"):
                filepath = os.path.join(source_dir, fileName)
                loader = UnstructuredMarkdownLoader(filepath)
                documents.extend(loader.load())

        texts = text_splitter.split_documents(documents=documents)
        print(f"Adding {len(documents)} documents chunks to the vector store...")
        vector_store = Chroma.from_documents(
            documents=texts,
            embedding=embeddingFn,
            persist_directory=db_path,
            collection_name=collection_name,
        )
        end_time = time.time()
        print(f"Processed MD files in {end_time - start_time:.2f} seconds")
        existing_ids = set(vector_store.get(include=[])["ids"])
        print(f"Total document chunks in store: {len(existing_ids)}")
    else:
        vector_store = Chroma(
            collection_name=collection_name,
            persist_directory=db_path,
            embedding_function=embeddingFn,
        )
        existing_ids = set(vector_store.get(include=[])["ids"])
        print(f"Total document chunks in store: {len(existing_ids)}")
    return vector_store.as_retriever(search_kwargs={"k": 5})
