# agent.py (REPLACE your existing agent.py with this)
import json
import os
import traceback

# Local modules from your repo
from database import insert_order, get_order_history, get_cart, cancel_order, save_cart
from rag import retrieve_relevant_menu, vectorstore
import utils.menu_matcher as menu_matcher
import utils.nlu as nlu_module

# Optionally use the same LLM backend only for free-form replies (smalltalk)
from langchain_groq import ChatGroq
from dotenv import load_dotenv
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

def _make_llm(temp: float = 0.2):
    key = GROQ_API_KEY
    try:
        return ChatGroq(model="openai/gpt-oss-20b", groq_api_key=key, temperature=temp)
    except Exception:
        return None

llm_singleton = _make_llm(0.2)


def _format_menu_text(raw_docs_text: str) -> str:
    """
    Takes the raw joined doc strings from retrieve_relevant_menu and formats bullet list.
    The raw docs are likely lines like: 'Beverages Coke Chilled Coca-Cola soda. $2.49'
    We'll be forgiving in parsing and produce a friendly list.
    """
    lines = [l.strip() for l in raw_docs_text.splitlines() if l.strip()]
    if not lines:
        return "No menu items matched your search."
    out = []
    for line in lines:
        # Try to partition into category / name / description / $price
        # We'll be conservative: put the line as-is if parsing fails
        try:
            # naive split on $ to pull price
            if "$" in line:
                left, price = line.rsplit("$", 1)
                price = "$" + price.strip()
            else:
                left, price = line, ""
            parts = left.split(None, 1)
            if len(parts) == 2:
                cat, rest = parts
            else:
                cat, rest = "", left
            out.append(f"- {rest.strip()} {price}".strip())
        except Exception:
            out.append(f"- {line}")
    return "Here are the matching menu items:\n" + "\n".join(out)


def _apply_save10_discount(total: float, promo_code: str | None) -> (float, float):
    discount = 0.0
    if promo_code and promo_code.upper() == "SAVE10":
        discount = total * 0.10
    return total - discount, discount


def process_query(query: str, session_id: str) -> str:
    """
    Deterministic controller:
    1. Use NLU to parse the user's utterance into an intent + structured items (if any)
    2. Perform the action using local functions (cart, db, retriever)
    3. Return a natural language response
    """
    try:
        parsed = nlu_module.nlu_extract(None, query, min_confidence=0.2)
    except Exception as e:
        parsed = {"intent": "help", "items": [], "confidence": 0.0}

    intent = parsed.get("intent", "help")
    items = parsed.get("items", []) or []
    promo = parsed.get("promo_code") or parsed.get("promo") or parsed.get("promo_code", "")
    # sanitize
    if isinstance(promo, str) and promo.strip() == "":
        promo = None

    # Intents handling
    if intent.startswith("menu"):
        # Compose query for retriever: if items present, use item name(s), else use raw query
        retriever_query = " ".join([it.get("name", "") for it in items]) if items else query
        docs_text = retrieve_relevant_menu(retriever_query, k=10)
        return _format_menu_text(docs_text)

    elif intent in ("order.add", "order.replace"):
        if not items:
            return "I didn't find any items in your message. Try 'Add 2 margherita pizzas' or 'Add coke'."
        # Normalize items against menu using utils.menu_matcher
        normalized = menu_matcher.normalize_items_against_menu(items)
        cart = get_cart(session_id)
        # merge into cart
        for n in normalized:
            found = next((c for c in cart if c["name"].lower() == n.name.lower()), None)
            unit_price = n.unit_price()
            if found:
                found["quantity"] = found.get("quantity", 0) + n.quantity
            else:
                cart.append({"name": n.name, "price": unit_price, "quantity": n.quantity})
        save_cart(session_id, cart)
        added_lines = [f"{n.quantity}x {n.name} (@ {n.unit_price():.2f} each)" for n in normalized]
        return "Added to cart: " + "; ".join(added_lines) + f". Current cart total: ${sum(i['price']*i['quantity'] for i in cart):.2f}"

    elif intent == "order.remove":
        if not items:
            return "Please tell me which item to remove (e.g., 'remove 1 coke')."
        cart = get_cart(session_id)
        if not cart:
            return "Your cart is empty."
        normalized = menu_matcher.normalize_items_against_menu(items)
        removed_lines = []
        for n in normalized:
            found = next((c for c in cart if c["name"].lower() == n.name.lower()), None)
            if found:
                remove_qty = min(found.get("quantity", 0), n.quantity)
                found["quantity"] -= remove_qty
                removed_lines.append(f"Removed {remove_qty} x {found['name']}")
                if found["quantity"] <= 0:
                    cart = [c for c in cart if c["name"].lower() != found["name"].lower()]
            else:
                removed_lines.append(f"{n.name} was not in your cart.")
        save_cart(session_id, cart)
        return "\n".join(removed_lines) + f"\nCart total now: ${sum(i['price']*i['quantity'] for i in cart):.2f}"

    elif intent == "order.confirm":
        cart = get_cart(session_id)
        if not cart:
            return "Your cart is empty. Add items before confirming an order."
        # Try to parse customer details if included in query
        # nlu may include customer_phone/address/name
        customer_name = parsed.get("customer_name")
        customer_phone = parsed.get("customer_phone")
        customer_address = parsed.get("customer_address")

        missing = []
        if not customer_name:
            missing.append("name")
        if not customer_phone:
            missing.append("phone")
        if not customer_address:
            missing.append("address")
        # If missing, ask the user to provide them (sidebar in app.py already has Save Details: they should save details there)
        if missing:
            return ("I need customer details before placing the order. Please save your Name, Phone (10 digits), and Address "
                    "in the 'Your Details' section on the sidebar, or include them in the message (e.g., 'Confirm order for Aditya, 9876543210, 123 My Street').")

        # Validate phone
        if not (isinstance(customer_phone, str) and customer_phone.isdigit() and len(customer_phone) == 10):
            return "Invalid phone number. Phone must be exactly 10 digits."

        total_price = sum(item["price"] * item["quantity"] for item in cart)
        total_after, discount = _apply_save10_discount(total_price, promo)
        # Insert order
        customer_details = {"name": customer_name, "phone": customer_phone, "address": customer_address}
        insert_order(session_id, cart, total_after, customer_details)
        # Clear cart
        save_cart(session_id, [])
        msg = f"Thank you {customer_name}! Your order has been placed. Total: ${total_after:.2f}."
        if discount > 0:
            msg += f" (SAVE10 discount applied: -${discount:.2f})"
        return msg

    elif intent == "order.cancel":
        # Try to extract an order id from the parsed fields or the query
        # NLU may put an order id; otherwise attempt to find integer in the query
        order_id = None
        try:
            order_id = int(parsed.get("order_id")) if parsed.get("order_id") else None
        except Exception:
            order_id = None
        if not order_id:
            # try to find an integer in the raw text
            import re
            m = re.search(r"\b(\d{1,7})\b", query)
            if m:
                order_id = int(m.group(1))
        if not order_id:
            return "Please specify the order id to cancel (for example 'Cancel order 123'). You can view pending orders from the 'Cancel Pending Order' section."

        # call cancel_order (database)
        res = cancel_order(order_id)
        return res

    elif intent == "history.show":
        history = get_order_history(session_id)
        if not history:
            return "You have no past orders."
        lines = []
        for o in history:
            items_str = ", ".join([f"{it['quantity']}x {it['name']}" for it in o['items']])
            lines.append(f"Order ID: {o['id']} | Status: {o['status']} | Total: ${o['total']:.2f} | Items: {items_str} | Date: {o['time']}")
        return "\n---\n".join(lines)

    elif intent == "smalltalk" or intent == "help":
        # Simple smalltalk: delegate to LLM if available for friendly phrasing
        if llm_singleton:
            try:
                prompt = f"User message: {query}\nRespond as a friendly assistant in one short paragraph."
                res = llm_singleton.invoke(prompt)
                # ChatGroq may return differently; try to extract text
                if isinstance(res, dict):
                    content = res.get("text") or res.get("content") or str(res)
                else:
                    content = str(res)
                return content
            except Exception:
                pass
        # fallback
        return ("I'm an ordering assistant. You can ask me to 'show menu', 'add 2 margherita', 'confirm order', "
                "'get my order history', or 'cancel order 123'. If you want to search the menu, try 'show beverages' or 'show pizzas'.")

    else:
        # Unknown intent fallback
        return ("Sorry — I couldn't interpret that. Try 'show menu', 'add 2 margherita', 'confirm order', "
                "'get my order history', or 'cancel order 123'.")
