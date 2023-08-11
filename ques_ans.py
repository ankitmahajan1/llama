from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
import chromadb
from llama_cpp import Llama
from uuid import uuid4
from langchain.chains.question_answering import load_qa_chain
from chromadb.config import Settings
from langchain.prompts import PromptTemplate
from langchain.llms.llamacpp import LlamaCpp
from langchain.schema import Document


# instantiate llama model
model_path = "./model/llama-2-7b-chat.ggmlv3.q4_0.bin"
llm_embedding = Llama(model_path=model_path, embedding=True)
llm_model = LlamaCpp(model_path=model_path, n_ctx=4000)

# instantiate chroma db 
COLLECTION_NAME='knowledge_base'
client = chromadb.Client(Settings(is_persistent=True))
collection = client.get_or_create_collection(COLLECTION_NAME)



def load_documents():
    loader = DirectoryLoader('./pdf', loader_cls=PyPDFLoader)
    documents = loader.load()

    print(f"number of files are {len(documents)}")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000)
    split_documents = text_splitter.split_documents(documents)

    print(f"spit documents are {len(split_documents)}")

    for chunk in split_documents:
        text_content = chunk.page_content
        # get embedding of chunk 
        try:
            result = llm_embedding.create_embedding(input=text_content)
            text_embedding = result['data'][0]['embedding']
            collection.add(
                embeddings=[text_embedding],
                documents=[text_content],
                ids=[str(uuid4())]
            )   
        except Exception as exp:
            print(f"exception raised for chunk: {chunk}")
            continue
    print("chunking complete")


def get_result(query):
    documents = []
    result = llm_embedding.create_embedding(input=query)
    query_embedding = result['data'][0]['embedding']

    # get chroma collection
    collection = client.get_or_create_collection(COLLECTION_NAME)
    docs = collection.query(query_embeddings=[query_embedding],
            n_results=3, include=["documents"])
    for doc in docs['documents'][0]:
        document = Document(page_content=doc)
        documents.append(document)
    qa_chain = load_qa_chain(llm=llm_model, chain_type="stuff")
    result = qa_chain.run(input_documents=documents, question=query)
    print(result)



query = "why old women transferred the property"
get_result(query)