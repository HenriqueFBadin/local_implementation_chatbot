from langchain_openai import AzureOpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
import os
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

embedder = AzureOpenAIEmbeddings(
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    azure_deployment=os.environ["AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME"],
    openai_api_version=os.environ["AZURE_OPENAI_EMBEDDING_API_VERSION"],
    api_key=os.environ["AZURE_OPENAI_EMBEDDING_API_KEY"],
)
store = Chroma(
    persist_directory="chroma",
    collection_name="example_collection",
    embedding_function=embedder,
)

SYSTEM_PROMPT_RAG_SOMENTE_MANUAIS = """
Você deve responder EXCLUSIVAMENTE com base no CONTEXTO abaixo (trechos de manuais recuperados).
Se algo não estiver no CONTEXTO, diga: "Não encontrado no manual fornecido." e peça os trechos/seções necessários.
Não use conhecimento externo. Não invente. A única exceção é quando o usuário menciona explicitamente que quer uma recomendação geral não relacionada com a empresa, aí você pode usar seu conhecimento.
"""

query = "Preciso compartilhar documentos confidenciais com meu time. O que eu faço?"
retriever = store.as_retriever(search_kwargs={"k": 10})

chat = AzureChatOpenAI(
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"],
    openai_api_version=os.environ["AZURE_OPENAI_API_VERSION"],
    api_key=os.environ["AZURE_OPENAI_API_KEY"],
    temperature=0.1,
)

PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT_RAG_SOMENTE_MANUAIS + "\n\nCONTEXT0:\n{context}"),
        ("human", "{input}"),
    ]
)

combine_docs_chain = create_stuff_documents_chain(chat, PROMPT)

rag_chain = create_retrieval_chain(retriever, combine_docs_chain)

result = rag_chain.invoke(
    {"input": query},
)

doc_n = 0
document_list = []

for doc in result["context"]:
    doc_n += 1
    nome = Path(doc.id.rsplit(":", 2)[0]).name
    document_list.append(f"[{doc_n}] {nome}\n")

resposta = result["answer"] + "\n\nDocumentos usados:\n" + "".join(document_list)

print("Pergunta:\n", query)
print("Resposta:\n", resposta)
print("\n--------------------------\n")