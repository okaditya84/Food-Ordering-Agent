# agent.py

from langchain_groq import ChatGroq
from langchain.agents import Tool, create_react_agent, AgentExecutor
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
import os
import json
import re

from database import insert_order, get_order_history, get_cart, cancel_order, save_cart
from rag import retrieve_relevant_menu

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
llm = ChatGroq(model="openai/gpt-oss-20b", groq_api_key=GROQ_API_KEY, temperature=0.5)

# Enhanced DB Interact Tool
def db_interact(input_str):
    try:
        data = json.loads(input_str)
        action = data.get("action")
        if action == "insert_order":
            session_id = data["session_id"]
            items = data["items"]
            total_price = sum(item["price"] * item["quantity"] for item in items)
            if data.get("promo") == "SAVE10":
                total_price *= 0.9
            customer_details = data.get("customer_details", {})
            phone = customer_details.get("phone", "")
            if len(phone) != 10 or not phone.isdigit():
                return "Invalid phone number. Must be exactly 10 digits (India format)."
            insert_order(session_id, items, total_price, customer_details)
            return f"Order inserted successfully. Total (after discount if applied): ${total_price:.2f}"
        elif action == "get_history":
            session_id = data["session_id"]
            history = get_order_history(session_id)
            return json.dumps(history)
        else:
            return "Invalid action"
    except Exception as e:
        return str(e)

db_tool = Tool(
    name="DB_Interact",
    func=db_interact,
    description="Use to insert orders or get history. Input must be a valid JSON string, e.g., '{\"action\": \"insert_order\", \"session_id\": \"id\", \"items\": [...], \"customer_details\": {...}, \"promo\": \"SAVE10\" optional}'. For insert_order, include 'items' list, 'session_id', 'customer_details': {'name': str, 'phone': str (10 digits), 'address': str}, optional 'promo':'SAVE10' for 10% discount."
)

# Menu Retrieval Tool (Updated to Accept JSON Input)
def menu_retriever(input_str):
    try:
        data = json.loads(input_str)
        query = data["query"]
        return retrieve_relevant_menu(query)
    except Exception as e:
        return str(e)

menu_tool = Tool(
    name="Menu_Retriever",
    func=menu_retriever,
    description="Retrieve relevant menu items based on query. Use for finding item names, prices, descriptions. Input: valid JSON string, e.g., '{\"query\": \"pizza\"}' or '{\"query\": \"toppings for pizza\"}'."
)

# Enhanced Cart Interact Tool
def cart_interact(input_str):
    try:
        data = json.loads(input_str)
        action = data["action"]
        session_id = data["session_id"]
        cart = get_cart(session_id)
        if action == "add_item":
            item = data["item"]  # {"name": str, "price": float, "quantity": int}
            existing = next((i for i in cart if i['name'] == item['name']), None)
            if existing:
                existing['quantity'] += item.get('quantity', 1)
            else:
                cart.append(item)
            save_cart(session_id, cart)
            return f"Added {item['quantity']} {item['name']} to cart."
        elif action == "remove_item":
            name = data["name"]
            quantity = data.get("quantity", 1)
            existing = next((i for i in cart if i['name'] == name), None)
            if existing:
                existing['quantity'] -= quantity
                if existing['quantity'] <= 0:
                    cart = [i for i in cart if i['name'] != name]
                save_cart(session_id, cart)
                return f"Removed {quantity} {name} from cart."
            return "Item not found in cart."
        elif action == "clear_cart":
            save_cart(session_id, [])
            return "Cart cleared successfully."
        elif action == "get_cart":
            return json.dumps(cart)
        else:
            return "Invalid action"
    except Exception as e:
        return str(e)

cart_tool = Tool(
    name="Cart_Interact",
    func=cart_interact,
    description="Manage user cart. Input must be a valid JSON string, e.g., '{\"action\": \"add_item\", \"session_id\": \"id\", \"item\": {\"name\": \"Pizza\", \"price\": 12.99, \"quantity\": 1}}'. Actions: add_item (with 'item' dict: {'name', 'price', 'quantity'}), remove_item ('name', optional 'quantity'), clear_cart, get_cart. Always include 'session_id'."
)

# Cancel Order Tool (Updated to Accept JSON Input)
def cancel_order_tool(input_str):
    try:
        data = json.loads(input_str)
        order_id = int(data["order_id"])
        return cancel_order(order_id)
    except Exception as e:
        return str(e)

cancel_tool = Tool(
    name="Cancel_Order",
    func=cancel_order_tool,
    description="Cancel a pending order by order_id. Input: valid JSON string, e.g., '{\"order_id\": \"123\"}'."
)

tools = [db_tool, menu_tool, cart_tool, cancel_tool]

# Enhanced Custom Prompt with Consistent JSON Input for All Tools
PREFIX = """You are a professional, helpful food ordering assistant. Handle natural language queries intelligently, including multi-item orders like '2 pizzas and a burger', menu browsing, cart management, customer details collection, order confirmation, cancellation, and storage. 

Key Guidelines:
- For menu queries (e.g., 'show pizzas'), use Menu_Retriever to fetch relevant items and describe them naturally based only on retrieved data. Do not invent items, toppings, or details not in the retrieval; if no specific toppings are retrieved, inform the user that customizable toppings are not available and suggest items as per menu descriptions.
- For adding items (e.g., 'add 2 Margherita Pizza'), parse quantities and names, use Menu_Retriever to confirm exact name/price if unsure, then use Cart_Interact add_item with retrieved details.
- For removing/clearing, use Cart_Interact remove_item or clear_cart.
- Always check cart with Cart_Interact get_cart before summarizing.
- Suggest items based on history: Use DB_Interact get_history to check past orders, suggest popular ones (e.g., 'You ordered X before, add again?').
- Collect customer details (name, 10-digit phone, address) before finalizing. Ask if missing. Validate phone strictly.
- Before inserting order, summarize cart, total (apply promo if mentioned), details, and ask for explicit confirmation (e.g., 'Confirm? Yes/No').
- If user confirms, use DB_Interact insert_order with items from cart, customer_details, and 'promo':'SAVE10' if user mentioned SAVE10 for 10% discount.
- After insert, clear cart via Cart_Interact.
- Handle promo: Only SAVE10, inform user of discount.
- For history: Use DB_Interact get_history and present nicely.
- For cancellation: Check if pending via history, then use Cancel_Order.
- Be conversational, polite, handle errors gracefully, suggest alternatives if item not found. Stick strictly to retrieved information; do not assume or invent additional options like toppings unless explicitly in menu retrieval.
- IMPORTANT: For ALL tools, input is a valid JSON string as specified in each tool's description. Ensure the Action Input is exactly a JSON string without extra quotes or escapes.

You have access to the following tools:"""

FORMAT_INSTRUCTIONS = """Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do step by step
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action (exactly a JSON string as per tool description; no extra text)
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question, phrased naturally and based only on verified information"""

template = PREFIX + "\n\n{tools}\n\n" + FORMAT_INSTRUCTIONS + "\n\nSession ID: {session_id}\n{chat_history}\nQuestion: {input}\nThought: {agent_scratchpad}"

prompt = PromptTemplate(
    input_variables=["session_id", "chat_history", "input", "agent_scratchpad", "tools", "tool_names"],
    template=template
)

agent = create_react_agent(llm, tools, prompt)

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=False)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors="Please check your output format carefully and try again. Ensure Action Input is exactly a valid JSON string as required.",
    max_iterations=15
)

def process_query(query, session_id):
    chat_history = memory.load_memory_variables({})["chat_history"]
    
    inputs = {
        "input": query,
        "session_id": session_id,
        "chat_history": chat_history,
        "agent_scratchpad": ""
    }
    
    try:
        response = agent_executor.invoke(inputs)
    except Exception as e:
        return f"Error processing request: {str(e)}. Please rephrase and try again."
    
    memory.save_context({"input": query}, {"output": response["output"]})
    
    return response["output"]