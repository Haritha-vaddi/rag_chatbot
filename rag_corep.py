

import os
import json
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import google.generativeai as genai





key = "xxxxxxxxxxxxxxxxxxxxxxxxxxxx"
PDF_PATH = Path("rag_project") / "Chapters 11 & 12  - Disclosures and Reporting.pdf"
CHROMA_DB_DIR = "chroma_corep_db1"
llm_model = "gemini-2.5-flash"


embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)




def insert_data_into_vectordb():
    """Load PDF, chunk it, and add to ChromaDB (run only once)"""
    
    print("\n" + "="*60)
    print("📥 INSERTING DATA INTO VECTOR DB")
    print("="*60)
    
   

   
    if not Path(PDF_PATH).exists():
        print(f"❌ PDF file not found at: {PDF_PATH}")
        print("Please provide a valid PDF path or create a sample PDF.")
        return False

    try:
        loader = PyPDFLoader(PDF_PATH)
        documents = loader.load()
        if not documents:
            print("❌ PDF loaded but contains no content.")
            return False
        print(f"✅ Loaded {len(documents)} pages from PDF")
    except Exception as e:
        print(f"❌ Error loading PDF: {e}")
        return False

   

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    chunks = text_splitter.split_documents(documents)

    

    print("⏳ Creating vector store...")
    vectorstore = Chroma(
        collection_name="corep_rules",
        embedding_function=embeddings,
        persist_directory=CHROMA_DB_DIR
    )

    vectorstore.add_documents(chunks)
    vectorstore.persist()
    print(f"✅ Added {len(chunks)} chunks to vector store")
    
    return True




def query_vectordb(query):
    """Retrieve documents and generate response using Gemini"""
    
    print("\n" + "="*60)
    print(f"🔍 QUERY: {query}")
    print("="*60)
    
   
    
    print("⏳ Loading vector store...")
    vectorstore = Chroma(
        collection_name="corep_rules",
        embedding_function=embeddings,
        persist_directory=CHROMA_DB_DIR
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    docs = retriever.invoke(query)
    context_text = "\n\n".join(doc.page_content for doc in docs)
    print(f"✅ Retrieved {len(docs)} relevant documents")

   

    print("⏳ Generating response from Gemini...")
    try:
        system_prompt_template = """You are an LLM-assisted PRA COREP reporting assistant.

Return STRICT JSON:
{
  "template": "COREP",
  "fields": [
    {
      "field_name": "",
      "value": "",
      "rule_reference": ""
    }
  ],
  "validation_flags": [],
  "audit_log": []
}"""
        
        user_prompt_template = f"""Context:
{context_text}

Question:
{query}"""
        
        model = genai.GenerativeModel(llm_model)
        resp = model.generate_content(
            system_prompt_template + "\n\n" + user_prompt_template
        )
        
        response_text = resp.text
        print("\n✅ Response received:")
        print("="*60)
        try:
           
            parsed = json.loads(response_text)
            print(json.dumps(parsed, indent=2))
        except json.JSONDecodeError:
           
            print(response_text)
        print("="*60 + "\n")
    except Exception as e:
        print(f"❌ Error generating response: {e}")




if __name__ == "__main__":
    
   
    db_exists = Path(CHROMA_DB_DIR).exists()
    
    if not db_exists:
        print("⚠️  Vector DB not found. Inserting data...")
        insert_data_into_vectordb()
    else:
        print("✅ Vector DB already exists. Skipping data insertion.")
    
   
    print("\n" + "="*60)
    query = input("📝 Enter your query: ").strip()
    print("="*60)
    
    if query:
        query_vectordb(query)
    else:
        print("❌ No query provided.")
