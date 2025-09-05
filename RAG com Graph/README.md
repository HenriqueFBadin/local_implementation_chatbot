# LangGraph RAG (Azure + Chroma) — Estrutura Modular

Este projeto demonstra um agente **RAG** com **LangGraph**, **Azure OpenAI** e **Chroma**, organizado em arquivos separados:
- `chatbot_main.py`: fluxo principal — monta o grafo, gera o diagrama e roda a conversa.
- `state_def.py`: estado compartilhado (histórico de mensagens com `add_messages`).
- `conditions.py`: função condicional `should_continue` (desvia para tools quando o LLM pede).
- `agents/llm_agent.py`: nó do LLM (injeta system prompt e usa tool-calling).
- `agents/tool_executor.py`: nó executor de ferramentas (executa `tool_calls` e devolve `ToolMessage`s).
- `tools/retrieve_documents.py`: vetor (Chroma), embeddings (Azure) e a tool `retrieve_documents` (+ helpers de ingestão).

## 1) Requisitos

Python 3.10+ recomendado. Instale as dependências:
```bash
pip install langgraph langchain langchain-openai langchain-chroma langchain-community langchain-text-splitters python-dotenv pandas pypdf
```

## 2) Variáveis de ambiente (.env)

Crie um `.env` na **raiz** com valores reais:

```env
AZURE_OPENAI_ENDPOINT=https://<seu-recurso>.openai.azure.com/
AZURE_OPENAI_API_VERSION=2024-xx-xx-preview
SECRET_AZURE_OPENAI_API_KEY=<sua-chave-chat>

AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME=<deployment-embeddings>
AZURE_OPENAI_EMBEDDING_API_VERSION=2024-xx-xx-preview
SECRET_AZURE_OPENAI_EMBEDDING_API_KEY=<sua-chave-embeddings>

AZURE_OPENAI_MODEL_DEPLOYMENT_NAME=<deployment-chat>  # ex.: gpt-4o-mini
```

> `AZURE_OPENAI_ENDPOINT` deve começar com `https://`.

## 3) Ingestão de dados (se a base estiver vazia)

O Chroma usa `persist_directory="chroma"` e `collection_name="example_collection"`.  
Se estiver vazio, a busca não retorna contexto. Para indexar:

```python
from tools.retrieve_documents import ingest_pdf, ingest_csv

ingest_pdf("data/Manual do OEE - TEEP V6.pdf")
ingest_csv("data/Perguntas_e_respostas_parcial.csv", text_columns="Perguntas,Respostas")
```
Isso persiste no diretório `chroma/`.

## 4) Rodando o chatbot

```bash
python chatbot_main.py
```
Exemplo de pergunta: `O que é OEE?`

## 5) Diagrama do grafo

- `graph.mmd` (Mermaid) sempre deve ser gerado.
- `graph.png` pode exigir Mermaid CLI:
  ```bash
  npm install -g @mermaid-js/mermaid-cli
  mmdc -i graph.mmd -o graph.png
  ```

## 6) Dicas

- Ajuste o system prompt em `agents/llm_agent.py`.
- Ajuste `k` do retriever e os `chunk_size/overlap` nos helpers de ingestão.
- Se quiser, troque o executor por `ToolNode` para reduzir código.
