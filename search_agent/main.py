from dotenv import load_dotenv
from typing import List
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langchain_tavily import TavilySearch

from pydantic import BaseModel, Field

_ = load_dotenv()


class Source(BaseModel):
    """Schema for a source used by the agent"""

    url: str = Field(description="The URL of the source")

class AgentResponse(BaseModel):
    """Schema for the agent response with answer and sources"""

    answer: str = Field(description="The agent's answer to the query")
    sources: List[Source] = Field(default_factory=list, description="List of sources used to generate the answer")

llm = ChatOpenAI(temperature=0, model="gpt-5")
tools = [TavilySearch()]
agent = create_agent(model=llm, tools=tools, response_format=AgentResponse)

def main():


    result = agent.invoke({
        "messages": HumanMessage(content="search for 3 job postings for an ai engineer using langchain in Netherlands on linkedin and list their details?")
    })

    print(result)


if __name__ == "__main__":
    main()
