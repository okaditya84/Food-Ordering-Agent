# app.py (Enhanced Streamlit Front-End with Better Integration)

import streamlit as st
import uuid
from agent import process_query
from rag import vectorstore
import json
from database import save_cart, get_cart, get_order_history
from datetime import datetime, timedelta
import time

st.title("AI Food Order Chatbot - Professional Edition")

# Session Management
if 'session_id' not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
    st.session_state.cart = get_cart(st.session_state.session_id)
    st.session_state.last_cart_update = datetime.now()
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'customer_details' not in st.session_state:
    st.session_state.customer_details = {"name": "", "phone": "", "address": ""}
if 'promo_code' not in st.session_state:
    st.session_state.promo_code = ""

# Process and Display Function
def process_and_display(input_text):
    st.session_state.messages.append({"role": "user", "content": input_text})
    with st.chat_message("user"):
        st.markdown(input_text)
    
    with st.chat_message("assistant"):
        with st.spinner("Processing your request..."):
            response = process_query(input_text, st.session_state.session_id)
        st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})
    
    # Sync cart after agent action (e.g., after insert or update)
    st.session_state.cart = get_cart(st.session_state.session_id)
    st.session_state.last_cart_update = datetime.now()
    st.rerun()

# Cart Timeout Check (1 hour inactivity)
if (datetime.now() - st.session_state.last_cart_update) > timedelta(hours=1) and st.session_state.cart:
    st.session_state.cart = []
    save_cart(st.session_state.session_id, [])
    st.warning("Your cart was cleared due to 1 hour of inactivity.")

# Display Chat History
chat_container = st.container()
with chat_container:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# Sidebar: Menu Browser, Cart, Details, Cancel
with st.sidebar:
    st.header("Menu Browser & Quick Actions")
    query = st.text_input("Search Menu Items")
    if query:
        results = vectorstore.similarity_search(query, k=10)  # Increased k for better results
        for doc in results:
            st.write(doc.page_content)
    else:
        with open('menu.json', 'r') as f:
            menu = json.load(f)
        for category in menu:
            st.subheader(category['category'])
            for item in category['items']:
                col1, col2 = st.columns([3,1])
                with col1:
                    st.write(f"{item['name']} - ${item['price']:.2f}: {item['description']}")
                with col2:
                    if st.button(f"Add {item['name']}", key=f"add_{item['name']}"):
                        with st.spinner("Adding to cart..."):
                            cart = st.session_state.cart
                            existing = next((i for i in cart if i['name'] == item['name']), None)
                            if existing:
                                existing['quantity'] += 1
                            else:
                                cart.append({"name": item['name'], "quantity": 1, "price": item['price']})
                            st.session_state.cart = cart
                            save_cart(st.session_state.session_id, cart)
                            st.session_state.last_cart_update = datetime.now()
                            st.success(f"Added {item['name']} to your cart!", icon="✅")

    # Current Cart Display
    st.subheader("Your Cart")
    cart = st.session_state.cart
    if cart:
        subtotal = sum(item['price'] * item['quantity'] for item in cart)
        promo_discount = 0
        if st.session_state.promo_code == "SAVE10":
            promo_discount = subtotal * 0.1
        total = subtotal - promo_discount
        for item in cart:
            col1, col2, col3 = st.columns([2,1,1])
            with col1:
                st.write(f"{item['name']} x {item['quantity']} - ${item['price'] * item['quantity']:.2f}")
            with col2:
                if st.button(f"+", key=f"plus_{item['name']}"):
                    with st.spinner("Updating..."):
                        item['quantity'] += 1
                        save_cart(st.session_state.session_id, cart)
                        st.session_state.last_cart_update = datetime.now()
                        st.rerun()
            with col3:
                if st.button(f"-", key=f"minus_{item['name']}"):
                    with st.spinner("Updating..."):
                        item['quantity'] -= 1
                        if item['quantity'] <= 0:
                            cart = [i for i in cart if i['name'] != item['name']]
                        st.session_state.cart = cart
                        save_cart(st.session_state.session_id, cart)
                        st.session_state.last_cart_update = datetime.now()
                        st.rerun()
        st.write(f"**Subtotal: ${subtotal:.2f}**")
        if promo_discount > 0:
            st.write(f"**Promo Discount (SAVE10): -${promo_discount:.2f}**")
        st.write(f"**Total: ${total:.2f}**")
        promo_input = st.text_input("Promo Code", value=st.session_state.promo_code)
        if st.button("Apply Promo"):
            if promo_input.upper() == "SAVE10":
                st.session_state.promo_code = "SAVE10"
                st.success("Promo code applied! Tell the bot to include it in your order.")
            else:
                st.error("Invalid promo code.")
        if st.button("Clear Cart"):
            with st.spinner("Clearing..."):
                st.session_state.cart = []
                save_cart(st.session_state.session_id, [])
                st.session_state.last_cart_update = datetime.now()
                st.success("Cart cleared!", icon="✅")
                st.rerun()
        if st.button("Confirm Order"):
            if st.session_state.promo_code == "SAVE10":
                process_and_display("Confirm my current cart order with promo SAVE10")
            else:
                process_and_display("Confirm my current cart order")
    else:
        st.info("Your cart is empty. Browse the menu to add items!")

    # Customer Details Form
    st.subheader("Your Details")
    name = st.text_input("Name", value=st.session_state.customer_details['name'])
    phone = st.text_input("Phone (10 digits, India)", value=st.session_state.customer_details['phone'])
    address = st.text_area("Address", value=st.session_state.customer_details['address'])
    if st.button("Save Details"):
        if len(phone) != 10 or not phone.isdigit():
            st.error("Phone must be exactly 10 digits.")
        else:
            st.session_state.customer_details = {"name": name, "phone": phone, "address": address}
            st.success("Details saved! The bot can use them for orders.")

    # Order Cancellation
    st.subheader("Cancel Pending Order")
    orders = get_order_history(st.session_state.session_id)
    if orders:
        pending_ids = [o['id'] for o in orders if o['status'] == 'Pending']
        if pending_ids:
            selected_order = st.selectbox("Select Pending Order to Cancel", pending_ids)
            if st.button("Cancel Selected Order"):
                process_and_display(f"Cancel order {selected_order}")
        else:
            st.info("No pending orders to cancel.")
    else:
        st.info("No order history yet.")

# User Input
user_input = st.chat_input("Chat with the bot: e.g., 'Add 2 pizzas', 'Show menu', 'Confirm order'")
if user_input:
    process_and_display(user_input)

# Quick Buttons
if st.button("View My Order History"):
    process_and_display("Get my order history")
if st.button("Show Full Menu"):
    process_and_display("Show me the restaurant menu")
if st.button("Suggest Something"):
    process_and_display("Suggest items based on my history or popular ones")