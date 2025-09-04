from langchain_community.document_loaders import PyPDFDirectoryLoader, CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import AzureOpenAIEmbeddings
from langchain_chroma import Chroma
import os
from dotenv import load_dotenv

load_dotenv()

# ---------- Load documents from the "data" directory ----------
pdf_loader = PyPDFDirectoryLoader(
    path="data",
)
pdf_docs = pdf_loader.load()

csv_loader = CSVLoader(
    file_path="data/Perguntas_e_respostas_parcial.csv", encoding="utf-8"
)
csv_docs = csv_loader.load()

docs = pdf_docs + csv_docs
print(f"Loaded {len(docs)} pages of documents")
# ---------------------------------------------------------------

# ---------- Split documents into chunks of text ----------
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=80,
    length_function=len,
    is_separator_regex=False,
)

chunks = text_splitter.split_documents(docs)
print(f"Split into {len(chunks)} chunks of text")
# ---------------------------------------------------------


# ---------- Transforming chunks into Documents ----------
def calculate_chunk_ids(chunks):
    """
    Cria ids únicos por chunk no formato:
      source:page:chunk_index
    e grava em chunk.metadata["id"].
    """
    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id = f"{source}:{page}"

        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id
        chunk.metadata["id"] = chunk_id

    return chunks


chunks_with_ids = calculate_chunk_ids(chunks)
# ------------------------------------------------------------

# ---------- Create embeddings using Azure ----------
embeddings = AzureOpenAIEmbeddings(
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    azure_deployment=os.environ["AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME"],
    openai_api_version=os.environ["AZURE_OPENAI_EMBEDDING_API_VERSION"],
    api_key=os.environ["AZURE_OPENAI_EMBEDDING_API_KEY"],
)
print("Created Azure embeddings")
# ----------------------------------------------------

vector_store = Chroma(
    collection_name="example_collection",
    embedding_function=embeddings,
    persist_directory="chroma",
)

# --- obtém IDs existentes no DB (a API devolve um dict com "ids")
existing_items = vector_store.get(include=[])  # ids são sempre incluídos por padrão
existing_ids = set(existing_items.get("ids", []))
print(f"Number of existing documents in DB: {len(existing_ids)}")

# --- filtra apenas os chunks novos
new_chunks = [c for c in chunks_with_ids if c.metadata.get("id") not in existing_ids]

if len(new_chunks):
    print(f"👉 Adding new documents: {len(new_chunks)}")
    new_chunk_ids = [c.metadata["id"] for c in new_chunks]
    vector_store.add_documents(new_chunks, ids=new_chunk_ids)
    # garante persistência
    try:
        vector_store.persist()
    except Exception:
        # fallback: algumas versões exigem outro modo de persistir
        try:
            vector_store._client.persist()
        except Exception:
            pass
    print("✅ Add + persist concluídos")
else:
    print("✅ No new documents to add")

# --- verificação simples: conferir contagem depois de inserir
after = vector_store.get(include=[])
print(f"Number of documents in DB (after): {len(set(after.get('ids', [])))}")
# --------------------------------------------------------------------------------------------
