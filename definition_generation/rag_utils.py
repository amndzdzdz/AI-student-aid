"""
This utils file contains the utility functions for the RAG mechanism that can be used with the inference script.
"""

import pandas as pd
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from sentence_transformers import SentenceTransformer
from langchain.embeddings import HuggingFaceEmbeddings

def index_data(data_path:str, store=False):
    """
    This function receives the data, and a boolean store and generates the vector store for the RAG mechanism

    Args:
        - data_path (str): The path to the csv dataset
        - store (bool): A boolean that determines whether the vector store is stored or not

    Output:
        - The vector store
    """
    data = pd.read_csv(data_path)
    documents = data.apply(lambda row: " ".join(map(str, row)), axis=1).tolist()
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.from_texts(texts=documents, embedding=embedding_model)

    if store:
        vector_store.save_local("faiss_store")
    
    return vector_store

def create_prompt(template: str, term: str, context: str) -> str:
    """
    This function receives a prompt template, a technical term and retrieved context through RAG and generates a prompt.

    Args:
        - template (str): Prompt template with variable {term} and {context}
        - term (str): The technical term 
        - context (str): The retrieved context

    Output:
        - template (str): The finished template with inserted technical term and context
    """
    template = template.replace("{term}", term)
    template = template.replace("{context}", context)
    
    return template

def retrieve_documents(term: str, sentence: str, vector_store) -> str:
    """
    This function reveives the technical term, the sentence the term was found in and the vector store and
    searches documents from the vector store that are similar to the technical term and returns them, as well as the
    sentence as additional context.

    Args:
        - term (str): The technical term
        - sentence (str): The sentence that contains the technical term
        - vector store: The vector store

    Output:
        - context (str): The sentence as well as the retrieved documents
    """
    results = []
    docs = vector_store.similarity_search_with_score(term)
    results = [doc[0].page_content for doc in docs if doc[-1] < 1]
    results = " ".join(results)
    context = sentence + " " + results

    return context

def generate_rag_response(term: str, sentence: str, model, vector_store) -> str:
    """
    This function receives the technical term, the sentence that contains the technical term, the LLM, and the
    vector store and uses the model to generate a technical term description with additional context.

    Args:
        - term (str): The technical term
        - sentence (str): The sentence that contains the technical term
        - model: The LLM that will generate the term definition
        - vector_store: The vector store for RAG

    Output:
        - reponse (str): The generated definition
    """
    template = "Explain the term {term} briefly. Additional context: {context} Explain the term briefly."
    context = retrieve_documents(term, sentence, vector_store)
    prompt = create_prompt(template, term, context)

    response = model.predict(prompt=prompt, tech_term=prompt, retriever=vector_store)

    return response