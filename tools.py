import json
import re
from langchain.tools import tool
from database import get_cart, save_cart, insert_order, get_order_history, cancel_order
from rag import retrieve_relevant_menu

# --- Tool Definitions ---

# To make the tools more reliable, we use the @tool decorator, which
# automatically infers the schema from the function's docstring and type hints.

@tool
def get_menu(query: str) -> str:
    """
    Use this tool to answer questions about the restaurant's menu.
    It takes a user's query (e.g., "what pizzas do you have?") and returns
    relevant menu items and their descriptions.
    """
    return retrieve_relevant_menu(query)

@tool
def view_cart(session_id: str) -> str:
    """
    Use this tool to see the items currently in the user's shopping cart.
    It takes the user's session_id and returns a summary of the cart.
    """
    cart = get_cart(session_id)
    if not cart:
        return "The cart is currently empty."
    
    total_price = sum(item['price'] * item['quantity'] for item in cart)
    cart_summary = "\n".join([f"- {item['quantity']}x {item['name']} (${item['price']:.2f} each)" for item in cart])
    
    return f"Current cart:\n{cart_summary}\n\nTotal: ${total_price:.2f}"

@tool
def add_to_cart(session_id: str, item_name: str, quantity: int) -> str:
    """
    Use this tool to add one or more items to the shopping cart.
    It takes the session_id, the name of the item, and the quantity to add.
    """
    # A simple way to find the item's price from the menu file.
    # In a real-world app, this would query a database.
    with open('menu.json', 'r') as f:
        menu = json.load(f)
    
    item_price = None
    for category in menu:
        for item in category['items']:
            if item['name'].lower() == item_name.lower():
                item_price = item['price']
                break
        if item_price:
            break
            
    if item_price is None:
        return f"Sorry, I couldn't find an item named '{item_name}' on the menu."
        
    cart = get_cart(session_id)
    
    # Check if item already exists in cart to update quantity
    found = False
    for item in cart:
        if item['name'].lower() == item_name.lower():
            item['quantity'] += quantity
            found = True
            break
    
    if not found:
        cart.append({"name": item_name, "price": item_price, "quantity": quantity})
        
    save_cart(session_id, cart)
    return f"Successfully added {quantity}x {item_name} to the cart."

@tool
def remove_from_cart(session_id: str, item_name: str, quantity: int) -> str:
    """
    Use this tool to remove one or more items from the shopping cart.
    It takes the session_id, the name of the item, and the quantity to remove.
    """
    cart = get_cart(session_id)
    
    item_found_in_cart = False
    for item in cart:
        if item['name'].lower() == item_name.lower():
            item_found_in_cart = True
            item['quantity'] -= quantity
            if item['quantity'] <= 0:
                cart.remove(item)
            break
            
    if not item_found_in_cart:
        return f"Item '{item_name}' not found in the cart."
        
    save_cart(session_id, cart)
    return f"Successfully removed {quantity}x {item_name} from the cart."

@tool
def place_order(session_id: str, customer_name: str, customer_phone: str, customer_address: str, promo_code: str = "") -> str:
    """
    Use this tool to finalize the order and save it to the database.
    It requires all customer details: name, a valid 10-digit phone number, and address.
    A promo_code can be optionally applied.
    """
    # Validate phone number
    if not re.match(r"^\d{10}$", customer_phone):
        return "Invalid phone number. Please provide a valid 10-digit phone number."

    cart = get_cart(session_id)
    if not cart:
        return "The cart is empty. Please add items before placing an order."

    total_price = sum(item['price'] * item['quantity'] for item in cart)
    
    # Apply discount for promo code
    if promo_code.upper() == "SAVE10":
        discount = total_price * 0.10
        total_price -= discount
        
    customer_details = {
        "name": customer_name,
        "phone": customer_phone,
        "address": customer_address
    }
    
    insert_order(session_id, cart, total_price, customer_details)
    
    confirmation_message = f"Thank you, {customer_name}! Your order has been placed successfully. Total amount is ${total_price:.2f}."
    if promo_code.upper() == "SAVE10":
        confirmation_message += " (SAVE10 discount applied)."
        
    return confirmation_message

@tool
def get_user_order_history(session_id: str) -> str:
    """
    Use this tool to retrieve the past order history for the current user.
    It takes the session_id and returns a list of their previous orders.
    """
    history = get_order_history(session_id)
    if not history:
        return "You have no past orders."
    
    history_summary = []
    for order in history:
        items_str = ", ".join([f"{item['quantity']}x {item['name']}" for item in order['items']])
        summary = (
            f"Order ID: {order['id']} | Status: {order['status']} | Total: ${order['total']:.2f} | "
            f"Items: {items_str} | Date: {order['time']}"
        )
        history_summary.append(summary)
        
    return "\n---\n".join(history_summary)

@tool
def cancel_user_order(order_id: int) -> str:
    """
    Use this tool to cancel a user's order.
    It takes the order_id and can only cancel orders that are in 'Pending' status.
    """
    return cancel_order(order_id)