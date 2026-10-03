import os
from dotenv import load_dotenv

from langchain_unstructured import UnstructuredLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

_ = load_dotenv()

if __name__ == '__main__':
    print("Ingesting...")

    # max characters set to 1mn to have a single Document object created by the loader
    loader = UnstructuredLoader(file_path="./rag_gist/mediumblog1.txt", chunking_strategy="basic", max_characters=1000000)

    document = loader.load()

    print("splitting...")
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents=document)
    print(f"created {len(texts)} chunks")

    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

    print("ingesting...")
    PineconeVectorStore.from_documents(
        documents=texts,
        embedding=embeddings,
        index_name=os.environ.get("INDEX_NAME"),
    )

    print("finish")

