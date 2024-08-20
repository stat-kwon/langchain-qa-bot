import logging
import os
import sys

from dotenv import load_dotenv
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Pinecone
from langchain_openai import OpenAIEmbeddings

load_dotenv()
logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(filename)s:%(lineno)d %(message)s",
)
logger = logging.getLogger(__name__)


def initialize_vectorstore():
    index_name = os.environ["PINECONE_INDEX"]
    embeddings = OpenAIEmbeddings()
    return Pinecone.from_existing_index(index_name, embeddings)


if __name__ == "__main__":
    print(sys.argv)
    file_path = sys.argv[1]
    loader = PyPDFLoader(file_path)
    raw_docs = loader.load()

    logger.info(f"Loaded {len(raw_docs)} documents")

    text_splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=30)
    docs = text_splitter.split_documents(raw_docs)
    logger.info("Split {len(docs)} documents")

    vectorstore = initialize_vectorstore()
    vectorstore.add_documents(docs)
