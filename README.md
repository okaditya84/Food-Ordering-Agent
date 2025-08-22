# AI Chatbot for Food Orders

## Overview
This is a fully functional, professional-grade AI chatbot for placing food orders. It uses Streamlit for the UI, LangChain for agent-based conversation flow, Groq API for LLM inference (free tier), SQLite for database, and RAG for menu retrieval. The bot handles natural language queries intelligently, manages carts, applies promos, suggests items, and more.

## Features
- **Natural Language Ordering**: Handles queries like "Add 2 pizzas and a burger", multi-item, quantities.
- **Menu Browsing**: Search and display menu via RAG.
- **Cart Management**: Add/remove/clear items via chat or sidebar buttons.
- **Order Processing**: Collect details, confirm, apply SAVE10 promo (10% discount), store in DB.
- **History & Cancellation**: View history, cancel pending orders.
- **Suggestions**: Based on past orders.
- **Admin Dashboard**: Insights on sales, best-sellers, daily trends, order management.
- **Timeout**: Cart clears after 1 hour inactivity.
- **Validation**: Phone number (10 digits).

## Setup
1. Install dependencies (free):
```bash
pip install -r requirements.txt
```
2. Replace GROQ_API_KEY in agent.py with your free Groq key (sign up at console.groq.com).
3. Run the app:
```bash
streamlit run app.py
```
4. For admin dashboard:
```bash
streamlit run admin.py
```
5. Test: Chat with bot, add items, confirm orders, check admin.

## GitHub Repo
Upload to a public GitHub repo: https://github.com/yourusername/ai-food-chatbot (replace with your actual repo).

## Notes
- Uses free tools only: Groq (free API), HuggingFace (free embeddings), SQLite (local DB).
- No paid services or heavy downloads required (embeddings model downloads ~100MB once).
- Documentation: Code is self-documented; see prompt in agent.py for agent logic.
- To test: Use queries like "Show appetizers", "Add Spring Rolls", "Confirm order with SAVE10".