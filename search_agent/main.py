from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langchain_tavily import TavilySearch

_ = load_dotenv()




llm = ChatOpenAI(temperature=0, model="gpt-5")
tools = [TavilySearch()]
agent = create_agent(model=llm, tools=tools)

def main():


    result = agent.invoke({
        "messages": HumanMessage(content="search for 3 job postings for an ai engineer using langchain in Prague on linkedin and list their details?")
    })

    print(result)


if __name__ == "__main__":
    main()
