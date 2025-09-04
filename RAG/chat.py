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

SYSTEM_PROMPT = """
Você receberá DUAS ENTRADAS:
- Prompt do usuário: a solicitação original.
- Resposta da LLM: a resposta que foi dada para essa solicitação.

Sua tarefa é verificar a consistência e a completude da Resposta da LLM em relação ao Prompt do usuário, seguindo estas regras:

1) Se a Resposta da LLM já for suficiente e resolver o problema do usuário com base apenas nas informações contidas no Prompt do usuário, devolva EXATAMENTE a mesma Resposta da LLM (sem acrescentar nada).

2) Se a Resposta da LLM não resolver completamente o problema porque faltam informações no Prompt do usuário, liste de forma clara e objetiva quais INFORMAÇÕES ESTÃO FALTANDO. Apresente essas faltas como perguntas diretas para o usuário confirmar/complementar.

3) Se o Prompt do usuário já indicar um caminho claro (ex.: menciona um sistema, cenário ou contexto específico), mas ainda restarem dúvidas que impedem a melhor orientação, devolva a Resposta da LLM e ACRESCENTE PERGUNTAS ESPECÍFICAS para esclarecer os pontos faltantes.

4) Se a Resposta da LLM não for condizente com o problema descrito no Prompt do usuário (fora de contexto), responda com:
   - Uma frase curta solicitando mais clareza: "Poderia desenvolver de forma mais clara o problema?"
   - EM SEGUIDA, apresente 3–6 PERGUNTAS HIPOTÉTICAS que ajudem o usuário a indicar o rumo correto (por exemplo, propondo possíveis sistemas/cenários/etapas onde o erro pode ocorrer). As perguntas devem ser fechadas ou de confirmação, para o usuário validar ou refutar rapidamente o caminho.

FORMATO DA SAÍDA:
- Caso 1: devolver exatamente a Resposta da LLM.
- Caso 2: começar com uma frase curta indicando que faltam informações e listar as perguntas objetivas.
- Caso 3: devolver a Resposta da LLM seguida das perguntas específicas.
- Caso 4: devolver "Poderia desenvolver de forma mais clara o problema?" e, logo abaixo, as perguntas hipotéticas que orientem os possíveis caminhos.

-------------------------------
EXEMPLOS DE USO (GENÉRICOS)
-------------------------------

[Exemplo A — faltam informações]
Prompt do usuário: "Estou tendo erro para fazer login."
Resposta da LLM: "Tente redefinir sua senha e, se não funcionar, contate o suporte."

Saída esperada:
"A resposta está incompleta porque faltam informações no prompt do usuário.
Por favor, confirme:
- Em qual sistema/plataforma você está tentando fazer login?
- Qual é a mensagem de erro exata apresentada?
- Você já tentou redefinir a senha e limpar cache/cookies?
- Está usando autenticação em dois fatores? Se sim, em qual etapa ocorre o erro?"

[Exemplo B — resposta adequada]
Prompt do usuário: "No Sistema X, ao tentar entrar recebo 'Usuário inexistente'."
Resposta da LLM: "No Sistema X, esse erro ocorre quando o usuário não tem cadastro. Solicite ao RH a criação do acesso ou crie o usuário no portal administrativo seguindo o procedimento 4.2; depois disso, aguarde 15 minutos e tente novamente."

Saída esperada:
"No Sistema X, esse erro ocorre quando o usuário não tem cadastro. Solicite ao RH a criação do acesso ou crie o usuário no portal administrativo seguindo o procedimento 4.2; depois disso, aguarde 15 minutos e tente novamente."

[Exemplo C — caminho claro, mas ainda com lacunas]
Prompt do usuário: "No app móvel Y, o login falha com 'credenciais inválidas'."
Resposta da LLM: "Verifique se o e-mail está correto e tente redefinir a senha."

Saída esperada:
"Verifique se o e-mail está correto e tente redefinir a senha.
Para orientar melhor, responda:
- Você consegue acessar com as mesmas credenciais na versão web do Y?
- A falha ocorre em Wi-Fi e 4G/5G?
- O app Y está atualizado na última versão?
- Existe política de senha expirada no Y (quando foi a última troca)?"

[Exemplo D — resposta fora de contexto (com perguntas que sugerem caminhos)]
Prompt do usuário: "Não consigo acessar a conta no serviço Z, aparece 'Acesso bloqueado temporariamente'."
Resposta da LLM: "Para configurar impressoras de rede, instale o driver ABC e reinicie o spooler."

Saída esperada:
"Poderia desenvolver de forma mais clara o problema?
Vamos tentar restringir um pouco o problema para que eu possa te ajudar. Para entendermos melhor, poderia confirmar:
- O erro ocorre no serviço Z de autenticação (login) ou em outro sistema?
- A mensagem 'Acesso bloqueado temporariamente' surgiu após muitas tentativas de senha incorreta?
- Você está tentando acessar via web, app móvel ou desktop?
- Há política de bloqueio por tentativas ou por MFA no serviço Z?
- Consegue acessar outros serviços com a mesma conta?"

"""

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            SYSTEM_PROMPT,
        ),
        (
            "human",
            "O prompt do usuário foi: {input_usuario}; O prompt da LLM foi: {input_llm}",
        ),
    ]
)

chain = prompt | chat
result_complementar = chain.invoke(
    {
        "input_usuario": query,
        "input_llm": resposta,
    }
)

print("Resposta complementar:\n")
print(result_complementar.content)
print("\n--------------------------\n")