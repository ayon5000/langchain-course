import os
from dotenv import load_dotenv
from operator import itemgetter

_ = load_dotenv()

MODEL = "gpt-5.2"

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

print("Initializing components...")

embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
llm = ChatOpenAI(model=MODEL, temperature=0)
vectorstore = PineconeVectorStore(index_name=os.environ.get("INDEX_NAME"), embedding=embeddings)
retreiever = vectorstore.as_retriever(search_kwargs={"k":3})

prompt_template = ChatPromptTemplate.from_template(
    """Answer the question based only on the following context:

    {context}

    Question: {question}

    Provide a detailed answer:"""
)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def retriveal_chain_without_lecl(query: str):
    """
    Simple retrieval chain without LCEL.
    Manually retrieves documents, formats them, and generates a response.

    Limitations:
    - Manual step-by-step execution
    - No built-in streaming support
    - No async support without additional code
    - Harder to compose with other chains
    - More verbose and error-prone
    """

    # Step 1: Retrieve relevant documents
    documents = retreiever.invoke(input=query)

    # Step 2: Format documents into a string named 'context'
    context = format_docs(documents)

    # Step 3: Format the prompt with context and question
    rag_prompt = prompt_template.format_messages(
        context=context,
        question=query,
    )

    # Step 4: Invoke LLM with the formatted messages
    response = llm.invoke(rag_prompt)

    return response.content


# ============================================================================
# IMPLEMENTATION 2: With LCEL (LangChain Expression Language) - BETTER APPROACH
# ============================================================================
def create_retrieval_chain_with_lcel():
    """
    Create a retrieval chain using LCEL (LangChain Expression Language).
    Returns a chain that can be invoked with {"question": "..."}

    Advantages over non-LCEL approach:
    - Declarative and composable: Easy to chain operations with pipe operator (|)
    - Built-in streaming: chain.stream() works out of the box
    - Built-in async: chain.ainvoke() and chain.astream() available
    - Batch processing: chain.batch() for multiple inputs
    - Type safety: Better integration with LangChain's type system
    - Less code: More concise and readable
    - Reusable: Chain can be saved, shared, and composed with other chains
    - Better debugging: LangChain provides better observability tools
    """

    retriveal_chain = (
        RunnablePassthrough.assign(
            context = itemgetter("question") | retreiever | format_docs
        )
        | prompt_template
        | llm
        | StrOutputParser()
    )

    return retriveal_chain



if __name__ == "__main__":
    print("Retreiving...")

    # Query
    query = "What ia pinecone in machine learning?"

    # ========================================================================
    # Option 0: Raw invocation without RAG
    # ========================================================================

    print("\n" + "=" * 60)
    print("Implementation 0: Raw LLM Invocation (No RAG)")
    print("=" * 60)
    result_raw = llm.invoke(
        [
            HumanMessage(content=query)
        ]
    )
    print("\nAnswer:")
    print(result_raw.content)


    # ========================================================================
    # Option 1: Use implementation WITHOUT LCEL
    # ========================================================================

    
    print("\n" + "=" * 60)
    print("IMPLEMENTATION 1: Without LCEL")
    print("=" * 60)
    result_without_lecl = retriveal_chain_without_lecl(query)
    print("\nAnswer:")
    print(result_without_lecl)

    # ========================================================================
    # Option 2: Use implementation WITH LCEL (Better Approach)
    # ========================================================================
    print("\n" + "=" * 60)
    print("IMPLEMENTATION 2: With LCEL - Better Approach")
    print("=" * 60)
    print("Why LCEL is better:")
    print("- More concise and declarative")
    print("- Built-in streaming: chain.stream()")
    print("- Built-in async: chain.ainvoke()")
    print("- Easy to compose with other chains")
    print("- Better for production use")
    print("=" * 60)

    chain_with_lecl = create_retrieval_chain_with_lcel()
    result_with_lecl = chain_with_lecl.invoke(input={
        "question": query
    })
    print("\nAnswer:")
    print(result_with_lecl)