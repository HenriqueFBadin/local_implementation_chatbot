from typing import Annotated
import os
from langchain.chat_models import init_chat_model
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, ToolMessage, AIMessage
from langchain_openai import AzureOpenAIEmbeddings
from langgraph.checkpoint.memory import InMemorySaver
import re

load_dotenv()

# Mudar para Postgres ou outro em produção
memory = InMemorySaver()
memory_config = {"configurable": {"thread_id": "1"}}


class State(TypedDict):
    messages: Annotated[list, add_messages]


graph_builder = StateGraph(State)

llm = init_chat_model(
    "azure_openai:gpt-4o-mini",
    azure_deployment=os.environ["AZURE_OPENAI_MODEL_DEPLOYMENT_NAME"],
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    api_version=os.environ["AZURE_OPENAI_API_VERSION"],
    api_key=os.environ["SECRET_AZURE_OPENAI_API_KEY"],
)


embedder = AzureOpenAIEmbeddings(
    azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
    azure_deployment=os.environ["AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME"],
    openai_api_version=os.environ["AZURE_OPENAI_EMBEDDING_API_VERSION"],
    api_key=os.environ["SECRET_AZURE_OPENAI_EMBEDDING_API_KEY"],
)

vector_store = Chroma(
    persist_directory="chroma",
    collection_name="example_collection",
    embedding_function=embedder,
)

retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 7})


@tool
def retrieve_documents(query: str) -> str:
    """Retrieve relevant documents from the vector store."""

    docs = retriever.invoke(query)

    if not docs:
        return "No relevant information found in the documents."

    result = []

    for i, doc in enumerate(docs):
        result.append(f"Document {i+1}:\n{doc}\n")

    return "\n\n".join(result)


@tool
def rewind_state(query: str) -> str:
    """
    Adjusts the pointer to the checkpoint where len(messages) == N.
    Pass the user's text, e.g.: 'go back to 6' or 'go back to question 6'.
    """

    global memory_config

    n = re.search(r"\d+", query)

    if not n:
        return "REWIND_ERROR: informe um número, ex.: 'voltar para 6'."

    N = int(n.group())

    to_replay = None
    for state in graph.get_state_history(memory_config):
        if len(state.values["messages"]) == N:
            # We are somewhat arbitrarily selecting a specific state based on the number of chat messages in the state.
            to_replay = state

    if to_replay is not None:
        memory_config = to_replay.config
        return f"State rewound to {N} messages."

    return f"REWIND_ERROR: não foi possível encontrar um estado com {N} mensagens."


tools = [retrieve_documents, rewind_state]

llm_with_tools = llm.bind_tools(tools)

system_prompt_rag = (
    "Você é um assistente RAG. Sempre que a pergunta envolver fatos dos documentos, "
    "chame a ferramenta `retrieve_documents` antes de responder. Use SOMENTE o que vier em CONTEXT. "
    "Citações: COPIE EXATAMENTE a tag que antecede cada trecho no CONTEXT (ex.: [Manual_OEE_TEEP_V6.pdf p.2]). "
    "NÃO use palavras genéricas como 'arquivo'; preserve o nome exibido entre colchetes. "
    "Se o CONTEXT for insuficiente, diga que não encontrou.\n\n"
    "Se o usuário pedir para 'voltar' ou 'rewind' para um estado anterior, use a ferramenta rewind_state com o número apropriado de mensagens para retornar àquele estado. "
    "O usuário pode dizer coisas como 'voltar para 6' ou 'voltar para a pergunta 6', indicando o número de mensagens para reverter."
)


def should_continue(state: State):
    """Check if the last message contains tool calls."""
    result = state["messages"][-1]
    return hasattr(result, "tool_calls") and len(result.tool_calls) > 0


tools_dict = {our_tool.name: our_tool for our_tool in tools}


def chatbot(state: State):
    msg = [SystemMessage(content=system_prompt_rag)] + state["messages"]
    message = llm_with_tools.invoke(msg)
    # Because we will be interrupting during tool execution,
    # we disable parallel tool calling to avoid repeating any
    # tool invocations when we resume.
    assert len(message.tool_calls) <= 1
    return {"messages": [message]}


def take_action(state: State) -> State:
    """Execute tool calls from the LLM's response."""

    tool_calls = state["messages"][-1].tool_calls
    results = []
    for t in tool_calls:
        print(
            f"Calling Tool: {t['name']} with query: {t['args'].get('query', 'No query provided')}"
        )

        if not t["name"] in tools_dict:  # Checks if a valid tool is present
            print(f"\nTool: {t['name']} does not exist.")
            result = "Incorrect Tool Name, Please Retry and Select tool from List of Available tools."

        else:
            result = tools_dict[t["name"]].invoke(t["args"].get("query", ""))
            print(f"Result length: {len(str(result))}")

        # Appends the Tool Message
        results.append(
            ToolMessage(tool_call_id=t["id"], name=t["name"], content=str(result))
        )

    print("Tools Execution Complete. Back to the model!")
    return {"messages": results}


# The first argument is the unique node name
# The second argument is the function or object that will be called whenever
# the node is used.
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_node("retriever_agent", take_action)
graph_builder.add_edge(START, "chatbot")
graph_builder.add_conditional_edges(
    "chatbot",
    should_continue,
    {
        True: "retriever_agent",
        False: END,
    },
)
graph_builder.add_edge("retriever_agent", "chatbot")
graph_builder.add_edge("chatbot", END)
graph = graph_builder.compile(checkpointer=memory)

try:
    mermaid_src = graph.get_graph().draw_mermaid()
    with open("graph.mmd", "w", encoding="utf-8") as f:
        f.write(mermaid_src)
    print("Mermaid salvo em graph.mmd")
except Exception:
    pass


def stream_graph_updates(user_input: str):
    for event in graph.stream(
        {"messages": [("user", user_input)]}, config=memory_config, stream_mode="values"
    ):
        msg = event["messages"][-1]
        if isinstance(msg, AIMessage) or getattr(msg, "type", None) == "ai":
            print("Assistant:", msg.content)


while True:
    try:
        user_input = input("User: ")
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break
        print("Input received. Processing...")
        stream_graph_updates(user_input)

    except:
        print("Input not available. Using default input for demonstration.")
        # fallback if input() is not available
        user_input = "What do you know about LangGraph?"
        print("User: " + user_input)
        stream_graph_updates(user_input)
        break
