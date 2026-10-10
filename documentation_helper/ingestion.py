import asyncio

import os
import ssl
from dotenv import load_dotenv
import certifi

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_tavily import TavilyCrawl, TavilyExtract, TavilyMap
from logger import (Colors, log_error, log_header, log_info, log_success, log_warning)

_ = load_dotenv()

# Configure SSL context to use certifi certificates
ssl_context = ssl.create_default_context(cafile=certifi.where())
os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

# Create an OpenAIEmbeddings instance using the "text-embedding-3-small" model.
# - show_progress_bar=False → disables progress bar display during embedding generation.
# - chunk_size=50 → processes text in chunks of 50 items at a time for efficiency. (i.e. 50 Document objects at a time)
# - retry_min_seconds=10 → waits at least 10 seconds before retrying if a request fails.
# This object is used to convert text into numerical vector embeddings for tasks like
# semantic search, clustering, or similarity comparison.
embeddings = OpenAIEmbeddings(model="text-embedding-3-small",
                              show_progress_bar=False,
                              chunk_size=50,
                              retry_min_seconds=10)

vectorstore = PineconeVectorStore(index_name=os.environ.get("INDEX_NAME"), embedding=embeddings)
tavily_crawl = TavilyCrawl()
tavily_extract = TavilyExtract()
tavily_map = TavilyMap(max_depth=5, max_breadth=20, max_pages=1000)





async def main():
    """Main async function to orchestrate the entire process."""
    log_header("DOCUMENTATION INGESTION PIPELINE")

    log_info(
        "🗺️  TavilyCrawl: Starting to crawl the documentation site",
        Colors.PURPLE,
    )
    # Crawl the documentation site

    res = tavily_crawl.invoke(
        {
            "url": "https://python.langchain.com/",
            "max_depth": 2,
            "extract_depth": "advanced",
        }
    )

    # Convert Tavily crawl results to LangChain Document objects
    all_docs = []
    for tavily_crawl_result_item in res["results"]:
        log_info(
            f"TavilyCrawl: Successfully crawled {tavily_crawl_result_item['url']} from documentation site"
        )

        if tavily_crawl_result_item["raw_content"] is not None:
            all_docs.append(
                Document(
                    page_content=tavily_crawl_result_item["raw_content"],
                    metadata={"source": tavily_crawl_result_item["url"]},
                )
            )



if __name__ == "__main__":
    asyncio.run(main())