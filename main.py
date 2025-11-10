import os
from dotenv import load_dotenv

from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import PromptTemplate

load_dotenv()

if __name__ == "__main__":
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    llm = ChatOllama(model="gemma3:1b")

    query = "what is Pinecone in machine learning?"

    chain = PromptTemplate.from_template(template=query) | llm
    # result = chain.invoke(input={})
    # print(result.content)


    # vectorstore
    vectorstore = PineconeVectorStore(
        index_name=os.environ["INDEX_NAME"],
        embedding=embeddings,
    )
    retriever = vectorstore.as_retriever()

    # Prompt
    rag_prompt = PromptTemplate.from_template("""
        Use the context to answer the question.
        If the answer is not found, say "No information in context".

        QUESTION: {input}

        CONTEXT:
        {context}
    """)

    # LCEL RAG Chain (CORRECT)
    chain = (
        {
            "input": lambda x: x["input"],
            "context": lambda x: retriever.invoke(x["input"]),
        }
        | rag_prompt
        | llm
    )

    result = chain.invoke({"input": query})

    print("\n=== ANSWER ===")
    print(result)
