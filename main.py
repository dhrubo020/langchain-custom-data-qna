import csv
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import numpy as np
import pickle
import os

embedding = OllamaEmbeddings(model="nomic-embed-text")
PERSIST_PATH = "./faiss_products"
LLM_MODEL = "deepseek-r1:1.5b"  # or "mistral", "tinyllama"


def create_or_load_vector_store(documents, persist_path=PERSIST_PATH):
    index_file_path = os.path.join(persist_path, "index.faiss")
    pkl_file_path = os.path.join(persist_path, "index.pkl")

    if os.path.exists(index_file_path) and os.path.exists(pkl_file_path):
        print("[INFO] Loading existing FAISS vector store...")
        with open(f"{persist_path}.pkl", "rb") as f:
            stored = pickle.load(f)
        index = FAISS.load_local(persist_path, embeddings=embedding,allow_dangerous_deserialization=True)
        return index
    else:
        print("[INFO] Creating new FAISS vector store...")
        store = FAISS.from_documents(documents, embedding)
        store.save_local(persist_path)
        with open(f"{persist_path}.pkl", "wb") as f:
            pickle.dump({"index_name": "index"}, f)
        return store


def clean_text(text: str) -> str:
    return (
        text.strip()
        .replace(";", ",")         # Convert feature separators
        .replace("  ", " ")        # Remove double spaces
    )

# STEP 1: Load CSV and create Documents
def load_product_csv(csv_file):
    docs = []
    with open(csv_file, newline='', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            # Clean each field
            name = clean_text(row.get('Title', 'Unknown product'))
            price = clean_text(row.get('Price', 'Unknown price'))
            features = clean_text(row.get('Feature', ''))

            # Skip empty rows
            if not name or not features:
                continue

            content = (
                f"Product: {name}. "
                f"Price: {price}. "
                f"Features: {features}."
            )
            docs.append(Document(page_content=content))
    return docs


def main():
    docs = load_product_csv("products.csv")
    store = create_or_load_vector_store(docs)
    retriever = store.as_retriever(search_type="similarity", search_kwargs={"k": 10})

    print(f"[INFO] Vector store contains {len(docs)} documents.")

    prompt_template = PromptTemplate.from_template("""
    You are a helpful product assistant. Based on the following product information:

    {context}

    Answer the question: {question}
    Don't provide anything outside the above given information.
    """)

    llm = ChatOllama(model=LLM_MODEL, temperature=0.2, max_tokens=512,streaming=True)
    chain = prompt_template | llm | StrOutputParser()

    while True:
        query = input("Ask a question about products: ")
        if query.lower() in ["exit", "quit"]:
            print("Exiting...")
            break

        print(f"[INFO] Retrieving relevant documents for query: {query}")
        relevant_docs = retriever.invoke(query)
        print(f"[INFO] Found {len(relevant_docs)} relevant documents.")

        context = "\n".join([doc.page_content for doc in relevant_docs])
        response = chain.invoke({"context": context, "question": query})
        print("AI:", response)


if __name__ == "__main__":
    main()
