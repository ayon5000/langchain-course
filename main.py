from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

_ = load_dotenv()


def main():


    information = """
    Sydney Bernice Sweeney (born September 12, 1997)[1] is an American actress. She gained early recognition for her roles in Everything Sucks!, The Handmaid's Tale, and Sharp Objects in 2018. She received wider acclaim for her performances in the drama series Euphoria (2019–2026) and the first season of the anthology series The White Lotus (2021), both of which earned her nominations for Primetime Emmy Awards.[2][3]

    In film, Sweeney garnered critical acclaim for her performances in the drama film Reality (2023) and for her portrayal of professional boxer Christy Martin in the biopic Christy (2025), and has also appeared in the box office hits Anyone but You (2023) and The Housemaid (2025). Her other film credits include Once Upon a Time in Hollywood (2019), Madame Web (2024), and Immaculate (2024).

    Sweeney has appeared in various advertising campaigns, some sparking controversy.
    """

    summary_template = """
    ## Given the information about a person I want you to create:
    1. A short summary
    2. two interesting facts about them
    ### Information:
    {information}
    """

    # Note: PromptTemplate.from_template() automatically infers input_variables
    # from placeholders in the template string (e.g., {topic}). You cannot pass
    # input_variables explicitly here. If you need to define them manually,
    # use the PromptTemplate(...) constructor instead.

    summary_prompt_template = PromptTemplate(
        input_variables=["information"],
        template=summary_template,
    )

    llm = ChatOpenAI(temperature=0, model="gpt-5")

    chain = summary_prompt_template | llm

    response = chain.invoke(
        input={
            "information":information,
        },
    )

    print(response.content)


if __name__ == "__main__":
    main()
