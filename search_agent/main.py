from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from tavily import TavilyClient

_ = load_dotenv()

tavily = TavilyClient()

@tool
def search_tool(query: str) -> str:
    """
    Tool that searches over the internet
    Args:
        query: The query to search for
    Returns:
        The search result
    """

    print(f"Searching for {query}")

    return tavily.search(query=query)

llm = ChatOpenAI(temperature=0, model="gpt-5")
tools = [search_tool]
agent = create_agent(model=llm, tools=tools)

def main():


    result = agent.invoke({
        "messages": HumanMessage(content="search for 3 job postings for an ai engineer using langchain in the Kolkata on linkedin and list their details?")
    })

    print(result)


if __name__ == "__main__":
    main()
