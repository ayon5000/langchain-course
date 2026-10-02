import json
from dotenv import load_dotenv

_ = load_dotenv()

from openai import OpenAI
from langsmith import traceable


MAX_ITERATIONS = 10
MODEL = "gpt-5"

# --- Tools (LangChain @tool decorator) ---
# Gets removed. In order to trace the functions as a tool
# In langsmith 

@traceable(run_type="tool")
def get_product_price(product: str)-> float:
    """Look up the price of a product in the catalog."""
    print(f"    >> Executing get_product_price(product='{product}')")
    prices = {"laptop": 1299.99, "headphones": 149.95, "keyboard": 89.50}
    return prices.get(product, 0)

@traceable(run_type="tool")
def apply_discount(price: float, discount_tier: str) -> float:
    """Apply a discount tier to a price and return the final price.
    Available tiers: bronze, silver, gold."""
    print(f"    >> Executing apply_discount(price={price}, discount_tier='{discount_tier}')")
    discount_percentages = {"bronze": 5, "silver": 12, "gold": 23}
    discount = discount_percentages.get(discount_tier, 0)
    return round(price * (1 - discount / 100), 2)


# Difference 2: Without @tool, we must MANUALLY define the JSON schema for each function.
# This is exactly what LangChain's @tool decorator generates automatically
# from the function's type hints and docstring.

tools_for_llm = [
    {
        "type": "function",
        "function": {
            "name": "get_product_price",
            "description": "Look up the price of a product in the catalog.",
            "parameters": {
                "type": "object",
                "properties": {
                    "product": {"type": "string", "description": "The product name, e.g. 'laptop', 'headphones', 'keyboard'"},
                },
                "required": ["product"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "apply_discount",
            "description": "Apply a discount tier to a price and return the final price. Available tiers: bronze, silver, gold.",
            "parameters": {
                "type": "object",
                "properties": {
                    "price": {"type": "number",  "description": "The original price"},
                    "discount_tier": {
                        "type": "string",
                        "description": "The discount tier: 'bronze', 'silver', or 'gold'",
                    }
                },
                "required": ["price", "discount_tier"]
            }
        }
    }
]


# --- Helper: traced OpenAI call ---
# Difference 3: Without LangChain, we must manually trace LLM calls for LangSmith.

@traceable(name="OpenAI Chat", run_type="llm")
def openai_chat_traced(messages):
    client = OpenAI()
    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        tools=tools_for_llm
    )
    return response



@traceable(name="Open AI Agent Loop")
def run_agent(question: str):

    # Difference 4: Without LangChain, we must manually create the dictionary for tools
    # because Langchain's @tool decorator added the name attribute to the function
    
    tools_dict = {
        "get_product_price":get_product_price,
        "apply_discount":apply_discount,
    }

    print(f"Question: {question}")
    print("=" * 60)

    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful shopping assistant. "
                "You have access to a product catalog tool "
                "and a discount tool.\n\n"
                "STRICT RULES — you must follow these exactly:\n"
                "1. NEVER guess or assume any product price. "
                "You MUST call get_product_price first to get the real price.\n"
                "2. Only call apply_discount AFTER you have received "
                "a price from get_product_price. Pass the exact price "
                "returned by get_product_price — do NOT pass a made-up number.\n"
                "3. NEVER calculate discounts yourself using math. "
                "Always use the apply_discount tool.\n"
                "4. If the user does not specify a discount tier, "
                "ask them which tier to use — do NOT assume one."
            ),
        },
        {"role": "user", "content": question},
    ]
    

    for iteration in range(1, MAX_ITERATIONS+1):
        print(f"\n--- Iteration {iteration} ---")

        # Difference 5: OpenAI().chat.completions.create directly instead of llm_with_tools.invoke()
        ai_message = openai_chat_traced(messages)

        tool_calls = ai_message.choices[0].message.tool_calls

        if not tool_calls:
            print(f"\nFinal Answer: {ai_message.choices[0].message.content}")
            return ai_message.choices[0].message.content

        # Process only the FIRST tool call — force one tool per iteration
        tool_call = tool_calls[0]
        # Difference 6: Attribute access (.function.name) instead of dict access (.get("name"))
        tool_name = tool_call.function.name
        tool_args = json.loads(tool_call.function.arguments)
        tool_id = tool_call.id

        print(f"  [Tool Selected] {tool_name} with args: {tool_args} and id {tool_id}")

        # check if the tool_name suggested by the LLM is valid
        tool_to_use = tools_dict.get(tool_name)

        # raise error in case of an invalid tool
        if tool_to_use is None:
            raise ValueError(f"Tool '{tool_name}' not found")

        # Difference 7: Direct function call instead of tool.invoke()
        observation = tool_to_use(**tool_args)

        # print output of tool invoked
        print(f"  [Tool Result] {observation}")

        # add the ai_message and observation (as ToolMessage) to messages 
        # to maintain trace across each iteration
        messages.append({
            "role":"assistant",
            "tool_calls": tool_calls  # model's request to call the tool
        })
        messages.append({
            "role": "tool",
            "tool_call_id": tool_id,
            "content": str(observation)
        })

    # If all iterations exhausted, print error
    print("ERROR: Max iterations reached without a final answer")
    return None


if __name__ == "__main__":
    print("Hello Langchain Agent (.bind_tools)")
    print()
    result = run_agent("What is the price of a laptop after applyting a gold discount")