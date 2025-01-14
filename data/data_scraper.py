import os
import json

import chromadb

from firecrawl import FirecrawlApp
from dotenv import load_dotenv
from llama_index.core import Document
from llama_index.llms.openai import OpenAI
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import Settings
from llama_index.core import StorageContext
from llama_index.core import VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.openai import OpenAIEmbedding


load_dotenv()
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
FIRECRAWL_API_KEY = os.environ["FIRECRAWL_API_KEY"]

app = FirecrawlApp(api_key=FIRECRAWL_API_KEY)



website_list = [
    {
        "Category": "BSA Website",
        "Description": "Scouts BSA website that contains all infrormation about scouting",
        "Name": "Scouting.org",
        "URL": "https://scouting.org",
        "Limit": 9999
    },
    {
        "Category": "BSA Blog",
        "Description": "Scouting magazine's blog for articles about scouting.",
        "Name": "Scouting Magazine",
        "URL": "https://blog.scoutingmagazine.org",
        "Limit": 1000
    }
]

# index = VectorStoreIndex.from_documents(documents)
# query_engine = index.as_query_engine()

def crawl_websites(websites):
    # Crawl websites and handle responses
    url_response = {}
    crawl_per_min = 3  # Max crawl per minute

    # Track crawls
    crawled_websites = 0
    scraped_pages = 0
    for i, website_dict in enumerate(websites):
        url = website_dict.get('URL')
        print(f"Crawling: {url}")

        try:
            response = app.crawl_url(
                url,
                params={
                    'limit': website_dict.get('Limit'),  # Limit pages to scrape per site.
                    'scrapeOptions': {'formats': ['markdown', 'html']}
                }
            )
            crawled_websites += 1

        except Exception as exc:
            print(f"Failed to fetch {url} -> {exc}")
            continue

        # Store the scraped data and associated info in the response dict
        url_response[url] = {
            "scraped_data": response.get("data"),
            "csv_data": website_dict
        }
    
    with open('scout_information.json', 'w') as json_file:
        json.dump(url_response, json_file)


def create_vector_store():
    with open('scout_information.json') as file:
        data = json.load(file)

    documents = [Document(text=t) for t in data]

    llm = OpenAI(model="gpt-4o-mini")
    embed_model = OpenAIEmbedding(model="text-embedding-3-large")
    text_splitter = SentenceSplitter(chunk_size=512, chunk_overlap=30)

    Settings.llm = llm
    Settings.embed_model = embed_model
    Settings.text_splitter = text_splitter

    chroma_client = chromadb.PersistentClient(path="./data/scout_data")
    chroma_collection = chroma_client.create_collection("scout_data")

    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    index = VectorStoreIndex.from_documents(
        documents,
        embed_model=OpenAIEmbedding(model="text-embedding-3-large"),
        storage_context=storage_context,
        show_progress=True
    )

if __name__ == '__main__':
    crawl_websites(website_list)
    create_vector_store()
