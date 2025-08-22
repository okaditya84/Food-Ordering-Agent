import streamlit as st
import uuid
import json
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go


# Import core components
from core.ai_agent import process_query, get_agent_status, reset_agent_session
from database import (
    get_cart, get_order_history,
    get_sales_insights
)
from core.recommendation_engine import get_personalized_recommendations, analyze_user_preferences
from core.knowledge_retrieval import search_menu

# Page configuration
st.set_page_config(
    page_title="🍽️ Food Ordering Agent",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
    }
    
    .chat-message {
        padding: 1.2rem;
        border-radius: 12px;
        margin: 0.8rem 0;
        border-left: 4px solid #667eea;
        background: rgba(255,255,255,0.05);
        backdrop-filter: blur(10px);
    }
    
    .user-message {
        background: linear-gradient(135deg, #E3F2FD 0%, #BBDEFB 100%);
        border-left-color: #2196F3;
    }
    
    .assistant-message {
        background: linear-gradient(135deg, #E8F5E8 0%, #C8E6C9 100%);
        border-left-color: #4CAF50;
    }
    
    .cart-item {
        background: linear-gradient(135deg, #FFF8E1 0%, #FFECB3 100%);
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border: 1px solid #FFD54F;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    
    .recommendation-card {
        background: linear-gradient(135deg, #FFFFFF 0%, #F5F5F5 100%);
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 16px rgba(0,0,0,0.1);
        margin: 0.8rem 0;
        border-left: 4px solid #FF6B6B;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 4px 16px rgba(0,0,0,0.2);
    }
    
    .dietary-tag {
        background: linear-gradient(135deg, #E8F5E8 0%, #C8E6C9 100%);
        color: #2E7D32;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.85rem;
        margin: 0.2rem;
        display: inline-block;
        font-weight: 500;
    }
    
    .price-tag {
        background: linear-gradient(135deg, #FFF3E0 0%, #FFE0B2 100%);
        color: #E65100;
        padding: 0.3rem 0.8rem;
        border-radius: 8px;
        font-weight: bold;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .status-indicator {
        width: 12px;
        height: 12px;
        border-radius: 50%;
        display: inline-block;
        margin-right: 0.5rem;
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.5; }
        100% { opacity: 1; }
    }
    
    .status-online { background: linear-gradient(135deg, #4CAF50, #66BB6A); }
    .status-processing { background: linear-gradient(135deg, #FF9800, #FFB74D); }
    .status-error { background: linear-gradient(135deg, #F44336, #EF5350); }
    
    .professional-button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.8rem 1.5rem;
        border-radius: 8px;
        font-weight: 500;
        cursor: pointer;
        transition: all 0.3s ease;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    
    .professional-button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0,0,0,0.2);
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
def initialize_session():
    """Initialize session state variables"""
    if 'session_id' not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.cart = get_cart(st.session_state.session_id)
        st.session_state.last_cart_update = datetime.now()
    
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    if 'customer_details' not in st.session_state:
        st.session_state.customer_details = {"name": "", "phone": "", "address": ""}
    
    if 'dietary_preferences' not in st.session_state:
        st.session_state.dietary_preferences = []
    
    if 'ui_mode' not in st.session_state:
        st.session_state.ui_mode = "chat"
    
    if 'show_recommendations' not in st.session_state:
        st.session_state.show_recommendations = True
    
    if 'voice_enabled' not in st.session_state:
        st.session_state.voice_enabled = False

def render_header():
    """Render the professional header"""
    st.markdown("""
    <div class="main-header">
        <h1>🍽️ Food Ordering Agent</h1>
        <p>AI Powered Food Ordering Assistant with Intelligent Personalization</p>
    </div>
    """, unsafe_allow_html=True)

def render_agent_status():
    """Render professional agent status indicator"""
    try:
        status = get_agent_status(st.session_state.session_id)
        state = status.get('state', 'listening')
        
        status_colors = {
            'listening': 'status-online',
            'processing': 'status-processing',
            'tool_calling': 'status-processing',
            'responding': 'status-processing',
            'error': 'status-error'
        }
        
        status_text = {
            'listening': 'Ready to assist',
            'processing': 'Processing your request...',
            'tool_calling': 'Accessing information...',
            'responding': 'Generating response...',
            'error': 'System error'
        }
        
        color_class = status_colors.get(state, 'status-online')
        text = status_text.get(state, 'Ready')
        
        st.markdown(f"""
        <div style="text-align: right; margin-bottom: 1.5rem; padding: 1rem; background: rgba(255,255,255,0.1); border-radius: 8px;">
            <span class="status-indicator {color_class}"></span>
            <strong>AI Agent Status:</strong> {text}
        </div>
        """, unsafe_allow_html=True)
        
    except Exception:
        pass

def render_chat_interface():
    """Render the main chat interface"""
    st.subheader("💬 Chat with FoodAI")
    
    # Agent status
    render_agent_status()
    
    # Chat container with professional styling
    chat_container = st.container()
    
    with chat_container:
        # Display chat history
        for message in st.session_state.messages:
            message_class = "user-message" if message["role"] == "user" else "assistant-message"
            
            st.markdown(f"""
            <div class="chat-message {message_class}">
                <strong>{'👤 You' if message["role"] == "user" else '🤖 AI Assistant'}:</strong><br>
                {message["content"]}
            </div>
            """, unsafe_allow_html=True)
    
    # Input area
    col1, col2 = st.columns([4, 1])
    
    with col1:
        user_input = st.chat_input("Ask me anything about food, menu, orders, or get personalized recommendations...")
    
    with col2:
        if st.button("🎤 Voice", disabled=not st.session_state.voice_enabled, help="Voice input (Coming soon)"):
            st.info("Voice input feature coming soon!")
    
    # Process user input
    if user_input:
        process_user_input(user_input)
    
    # Professional quick action buttons
    st.markdown("### Quick Actions")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("📋 Browse Menu", key="menu_btn", help="Explore our complete menu"):
            process_user_input("Show me the complete restaurant menu with categories")
    
    with col2:
        if st.button("🎯 Get Recommendations", key="rec_btn", help="Get personalized recommendations"):
            process_user_input("Give me personalized recommendations based on my preferences")
    
    with col3:
        if st.button("🛒 View Cart", key="cart_btn", help="Check your current cart"):
            process_user_input("Show me my current cart with total")
    
    with col4:
        if st.button("📊 Order History", key="history_btn", help="View your order history"):
            process_user_input("Show me my complete order history")

def process_user_input(input_text: str):
    """Process user input and update chat"""
    # Add user message
    st.session_state.messages.append({"role": "user", "content": input_text})
    
    # Show professional processing indicator
    with st.spinner("🤖 AI is analyzing your request..."):
        try:
            # Process with enhanced agent
            response = process_query(input_text, st.session_state.session_id)
            
            # Add assistant response
            st.session_state.messages.append({"role": "assistant", "content": response})
            
            # Update cart if needed
            st.session_state.cart = get_cart(st.session_state.session_id)
            st.session_state.last_cart_update = datetime.now()
            
        except Exception as e:
            error_response = f"I apologize, but I encountered a technical issue: {str(e)}. Please try rephrasing your request or contact support if this continues."
            st.session_state.messages.append({"role": "assistant", "content": error_response})
    
    # Rerun to update display
    st.rerun()

def render_professional_sidebar():
    """Render professional sidebar with advanced features"""
    with st.sidebar:
        st.header("🎛️ Control Center")
        
        # UI Mode selector
        st.session_state.ui_mode = st.selectbox(
            "Interface Mode",
            ["chat", "menu_browser", "analytics"],
            format_func=lambda x: {
                "chat": "💬 Intelligent Chat",
                "menu_browser": "📖 Menu Explorer",
                "analytics": "📊 Analytics Dashboard"
            }[x]
        )
        
        st.divider()
        
        # Personalized recommendations
        if st.session_state.show_recommendations:
            render_personalized_recommendations()
        
        st.divider()
        
        # Professional cart management
        render_professional_cart()
        
        st.divider()
        
        # Customer profile
        render_customer_profile()
        
        st.divider()
        
        # Advanced settings
        render_advanced_settings()

def render_personalized_recommendations():
    """Render personalized recommendations with professional styling"""
    st.subheader("🎯 Personalized for You")
    
    try:
        # Get user preferences
        user_prefs = analyze_user_preferences(st.session_state.session_id)
        
        if user_prefs.get('total_orders', 0) > 0:
            # Show personalized recommendations
            with st.spinner("Generating personalized recommendations..."):
                recommendations = get_personalized_recommendations(
                    st.session_state.session_id,
                    context="sidebar recommendations",
                    dietary_prefs=st.session_state.dietary_preferences,
                    k=3
                )
            
            st.markdown(f"""
            <div class="recommendation-card">
                <h4>🌟 Recommended for You</h4>
                <p>{recommendations}</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            # Show popular items for new users
            st.markdown("""
            <div class="recommendation-card">
                <h4>🔥 Popular Choices</h4>
                <p>Welcome! Try our customer favorites: Margherita Pizza, Chicken Burger, and Fresh Lemonade to get started!</p>
            </div>
            """, unsafe_allow_html=True)
    
    except Exception as e:
        st.error(f"Recommendations temporarily unavailable: {str(e)}")

def render_professional_cart():
    """Render professional cart with advanced features"""
    st.subheader("🛒 Smart Cart")
    
    cart = st.session_state.cart
    
    if cart:
        # Cart summary with professional styling
        total = sum(item['price'] * item['quantity'] for item in cart)
        item_count = sum(item['quantity'] for item in cart)
        
        st.markdown(f"""
        <div class="metric-card">
            <h3>{item_count} Items</h3>
            <h2>${total:.2f}</h2>
            <p>Total Cart Value</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Cart items with professional controls
        for i, item in enumerate(cart):
            with st.container():
                st.markdown(f"""
                <div class="cart-item">
                    <strong>{item['name']}</strong><br>
                    <span class="price-tag">${item['price']:.2f} × {item['quantity']} = ${item['price'] * item['quantity']:.2f}</span>
                </div>
                """, unsafe_allow_html=True)
                
                col1, col2, col3 = st.columns([1, 1, 1])
                with col1:
                    if st.button("➕", key=f"add_{i}", help="Add one more"):
                        process_user_input(f"add 1 more {item['name']}")
                with col2:
                    if st.button("➖", key=f"remove_{i}", help="Remove one"):
                        process_user_input(f"remove 1 {item['name']}")
                with col3:
                    if st.button("🗑️", key=f"delete_{i}", help="Remove all"):
                        process_user_input(f"remove all {item['name']} from cart")
        
        # Professional cart actions
        st.markdown("### Cart Actions")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔍 Optimize Order", help="Get AI suggestions to improve your order"):
                process_user_input("analyze and optimize my current order for better value")
        
        with col2:
            if st.button("✅ Confirm Order", help="Proceed to checkout"):
                process_user_input("I want to confirm and place my order")
        
        if st.button("🧹 Clear Cart", type="secondary", help="Remove all items"):
            process_user_input("clear my entire cart")
    
    else:
        st.info("Your cart is empty. Start by asking for menu recommendations or browse our menu!")

def render_customer_profile():
    """Render professional customer profile management"""
    st.subheader("👤 Your Profile")
    
    # Customer details
    with st.expander("📝 Personal Information"):
        name = st.text_input("Full Name", value=st.session_state.customer_details.get('name', ''))
        phone = st.text_input("Phone Number", value=st.session_state.customer_details.get('phone', ''), help="10-digit phone number")
        address = st.text_area("Delivery Address", value=st.session_state.customer_details.get('address', ''))
        
        if st.button("💾 Save Information"):
            if len(phone) == 10 and phone.isdigit():
                st.session_state.customer_details = {
                    'name': name,
                    'phone': phone,
                    'address': address
                }
                st.success("✅ Information saved successfully!")
            else:
                st.error("❌ Please enter a valid 10-digit phone number")
    
    # Dietary preferences
    with st.expander("🥗 Dietary Preferences"):
        dietary_options = [
            "vegetarian", "vegan", "gluten_free", "dairy_free",
            "nut_free", "low_carb", "keto", "halal", "kosher"
        ]
        
        selected_prefs = st.multiselect(
            "Select your dietary preferences:",
            dietary_options,
            default=st.session_state.dietary_preferences,
            help="This helps us recommend suitable items"
        )
        
        if st.button("🔄 Update Preferences"):
            st.session_state.dietary_preferences = selected_prefs
            st.success("✅ Preferences updated!")
    
    # User insights
    try:
        user_prefs = analyze_user_preferences(st.session_state.session_id)
        if user_prefs.get('total_orders', 0) > 0:
            with st.expander("📊 Your Insights"):
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Total Orders", user_prefs.get('total_orders', 0))
                    st.metric("Price Sensitivity", f"{user_prefs.get('price_sensitivity', 0.5):.1%}")
                
                with col2:
                    if user_prefs.get('preferred_categories'):
                        st.write("**Favorite Categories:**")
                        for cat in user_prefs['preferred_categories'][:3]:
                            st.write(f"• {cat}")
    except Exception:
        pass

def render_advanced_settings():
    """Render advanced settings and controls"""
    st.subheader("⚙️ Advanced Settings")
    
    # UI preferences
    st.session_state.show_recommendations = st.checkbox(
        "Show Personalized Recommendations", 
        value=st.session_state.show_recommendations,
        help="Display AI-powered recommendations in sidebar"
    )
    
    st.session_state.voice_enabled = st.checkbox(
        "Enable Voice Input (Beta)",
        value=st.session_state.voice_enabled,
        help="Enable voice input functionality"
    )
    
    # Session management
    if st.button("🔄 Reset Session", help="Clear conversation history and reset AI context"):
        reset_agent_session(st.session_state.session_id)
        st.session_state.messages = []
        st.success("✅ Session reset successfully!")
    
    # Debug info
    with st.expander("🔧 System Information"):
        st.write(f"**Session ID:** {st.session_state.session_id[:8]}...")
        st.write(f"**Messages:** {len(st.session_state.messages)}")
        st.write(f"**Cart Items:** {len(st.session_state.cart)}")
        st.write(f"**Last Update:** {st.session_state.last_cart_update.strftime('%H:%M:%S')}")
        
        try:
            status = get_agent_status(st.session_state.session_id)
            st.json(status)
        except Exception:
            st.write("Status information unavailable")

def render_menu_browser():
    """Render professional menu browser"""
    st.header("📖 Intelligent Menu Explorer")
    
    # Search and filters
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        search_query = st.text_input("🔍 Search menu items...", placeholder="e.g., spicy pizza, healthy options, under $10")
    
    with col2:
        category_filter = st.selectbox("Category", ["All", "Appetizers", "Main Courses", "Desserts", "Beverages"])
    
    with col3:
        price_filter = st.selectbox("Price Range", ["All", "Under $10", "$10-$20", "Over $20"])
    
    # Search button
    if st.button("🔍 Search Menu") or search_query:
        if search_query:
            with st.spinner("Searching menu with AI..."):
                results = search_menu(
                    search_query,
                    session_id=st.session_state.session_id,
                    dietary_prefs=st.session_state.dietary_preferences
                )
                st.markdown(f"### Search Results\n{results}")
    
    # Display menu with professional styling
    try:
        # Try enhanced menu first, fallback to original
        menu_files = ["menu.json"]
        menu_data = None
        
        for menu_file in menu_files:
            try:
                with open(menu_file, "r") as f:
                    menu_data = json.load(f)
                break
            except FileNotFoundError:
                continue
        
        if not menu_data:
            st.error("Menu data not available")
            return
        
        for category in menu_data:
            if category_filter != "All" and category["category"] != category_filter:
                continue
            
            st.subheader(f"🍽️ {category['category']}")
            
            for item in category["items"]:
                # Price filter
                price = item.get("price", 0)
                if price_filter == "Under $10" and price >= 10:
                    continue
                elif price_filter == "$10-$20" and (price < 10 or price > 20):
                    continue
                elif price_filter == "Over $20" and price <= 20:
                    continue
                
                # Professional item card
                with st.container():
                    col1, col2 = st.columns([3, 1])
                    
                    with col1:
                        st.markdown(f"**{item['name']}** - ${item['price']:.2f}")
                        st.write(item['description'])
                        
                        # Dietary tags with professional styling
                        dietary_tags = item.get('dietary_tags', [])
                        if dietary_tags:
                            tags_html = " ".join([f'<span class="dietary-tag">{tag}</span>' for tag in dietary_tags])
                            st.markdown(tags_html, unsafe_allow_html=True)
                        
                        # Nutrition info
                        nutrition = item.get('nutrition', {})
                        if nutrition:
                            st.write(f"📊 {nutrition.get('calories', 'N/A')} cal | "
                                   f"{nutrition.get('protein', 'N/A')}g protein | "
                                   f"{nutrition.get('carbs', 'N/A')}g carbs")
                    
                    with col2:
                        if st.button(f"➕ Add {item['name']}", key=f"menu_add_{item['name']}", help="Add to cart"):
                            process_user_input(f"add {item['name']} to my cart")
                
                st.divider()
    
    except Exception as e:
        st.error(f"Menu loading error: {str(e)}")

def render_analytics():
    """Render professional analytics dashboard"""
    st.header("📊 Analytics Dashboard")
    
    try:
        # User analytics
        user_prefs = analyze_user_preferences(st.session_state.session_id)
        orders = get_order_history(st.session_state.session_id)
        
        if orders:
            # Metrics row
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Orders", len(orders))
            with col2:
                total_spent = sum(order['total'] for order in orders)
                st.metric("Total Spent", f"${total_spent:.2f}")
            with col3:
                avg_order = total_spent / len(orders) if orders else 0
                st.metric("Avg Order Value", f"${avg_order:.2f}")
            with col4:
                recent_orders = len([o for o in orders if datetime.fromisoformat(o['time']) > datetime.now() - timedelta(days=30)])
                st.metric("Orders (30 days)", recent_orders)
            
            # Order history chart
            order_dates = [datetime.fromisoformat(order['time']) if isinstance(order['time'], str) else order['time'] for order in orders]
            order_totals = [order['total'] for order in orders]
            
            fig = px.line(
                x=order_dates,
                y=order_totals,
                title="Your Order History",
                labels={'x': 'Date', 'y': 'Order Total ($)'}
            )
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Category preferences
            if user_prefs.get('preferred_categories'):
                categories = user_prefs['preferred_categories'][:5]
                fig = px.pie(
                    values=[1] * len(categories),
                    names=categories,
                    title="Your Favorite Categories"
                )
                fig.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # System analytics (if available)
        try:
            sales_data = get_sales_insights()
            
            if sales_data.get('best_selling'):
                st.subheader("🏆 Popular Items")
                best_items = sales_data['best_selling'][:10]
                
                fig = px.bar(
                    x=[item[1] for item in best_items],
                    y=[item[0] for item in best_items],
                    orientation='h',
                    title="Most Popular Menu Items",
                    labels={'x': 'Orders', 'y': 'Item'}
                )
                fig.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                )
                st.plotly_chart(fig, use_container_width=True)
        
        except Exception:
            st.info("System analytics unavailable")
    
    except Exception as e:
        st.error(f"Analytics error: {str(e)}")

def main():
    """Main application function"""
    # Initialize session
    initialize_session()
    
    # Render header
    render_header()
    
    # Render sidebar
    render_professional_sidebar()
    
    # Main content based on mode
    if st.session_state.ui_mode == "chat":
        render_chat_interface()
    elif st.session_state.ui_mode == "menu_browser":
        render_menu_browser()
    elif st.session_state.ui_mode == "analytics":
        render_analytics()
    
    # Professional footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 2rem;">
        <p>🤖 <strong>Groq AI Inference</strong> | 🍽️ <strong>By Aditya Jethani</strong></p>
        <p><em>Intelligent • Personalized • Efficient</em></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()