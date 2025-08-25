import json
import re
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

from langchain.tools import tool
from langchain_groq import ChatGroq

from config import GROQ_API_KEY
from database import (
    get_cart, save_cart, insert_order, get_order_history
)
from core.knowledge_retrieval import search_menu
from core.recommendation_engine import get_personalized_recommendations, analyze_user_preferences
from core.natural_language_processor import process_user_input

def extract_session_id_from_input(input_text: str) -> str:
    """Extract session_id from input text that contains 'Session ID: [session_id]'"""
    session_match = re.search(r'Session ID:\s*([a-f0-9\-]+)', input_text)
    if session_match:
        return session_match.group(1)
    return ""

@dataclass
class OrderOptimization:
    original_total: float
    optimized_total: float
    savings: float
    suggestions: List[str]
    reasoning: str

class SmartOrderManager:
    """Intelligent order management with AI reasoning"""
    
    def __init__(self):
        self.llm = self._create_llm()
        self.menu_data = self._load_menu_data()
    
    def _create_llm(self) -> Optional[ChatGroq]:
        try:
            return ChatGroq(
                model="openai/gpt-oss-20b",
                groq_api_key=GROQ_API_KEY,
                temperature=0.3,
                max_tokens=1024
            )
        except Exception:
            return None
    
    def _load_menu_data(self) -> List[Dict]:
        try:
            # Try enhanced menu first, fallback to original
            menu_files = ["menu.json"]
            for menu_file in menu_files:
                try:
                    with open(menu_file, "r", encoding="utf-8") as f:
                        return json.load(f)
                except FileNotFoundError:
                    continue
            return []
        except Exception:
            return []
    
    def find_menu_item(self, item_name: str) -> Optional[Dict]:
        """Find menu item with fuzzy matching"""
        item_name_lower = item_name.lower()
        
        # Exact match first
        for category in self.menu_data:
            for item in category.get('items', []):
                if item.get('name', '').lower() == item_name_lower:
                    item['category'] = category.get('category', '')
                    return item
        
        # Fuzzy match
        best_match = None
        best_score = 0
        
        for category in self.menu_data:
            for item in category.get('items', []):
                menu_item_name = item.get('name', '').lower()
                
                # Simple fuzzy matching
                if item_name_lower in menu_item_name or menu_item_name in item_name_lower:
                    score = len(set(item_name_lower.split()) & set(menu_item_name.split()))
                    if score > best_score:
                        best_score = score
                        best_match = item
                        best_match['category'] = category.get('category', '')
        
        return best_match
    
    def validate_dietary_restrictions(self, cart_items: List[Dict], dietary_prefs: List[str]) -> Dict[str, Any]:
        """Validate cart items against dietary restrictions"""
        violations = []
        warnings = []
        
        for item in cart_items:
            menu_item = self.find_menu_item(item['name'])
            if not menu_item:
                continue
            
            item_dietary_tags = [tag.lower() for tag in menu_item.get('dietary_tags', [])]
            item_allergens = [allergen.lower() for allergen in menu_item.get('allergens', [])]
            
            for pref in dietary_prefs:
                pref_lower = pref.lower()
                
                # Check for violations
                if pref_lower == 'vegetarian' and 'vegetarian' not in item_dietary_tags and 'vegan' not in item_dietary_tags:
                    violations.append(f"{item['name']} is not vegetarian")
                elif pref_lower == 'vegan' and 'vegan' not in item_dietary_tags:
                    violations.append(f"{item['name']} is not vegan")
                elif pref_lower == 'gluten_free' and 'gluten' in item_allergens:
                    violations.append(f"{item['name']} contains gluten")
                elif pref_lower == 'dairy_free' and 'dairy' in item_allergens:
                    violations.append(f"{item['name']} contains dairy")
                elif pref_lower.startswith('no_') and pref_lower[3:] in item_allergens:
                    violations.append(f"{item['name']} contains {pref_lower[3:]}")
        
        return {
            'violations': violations,
            'warnings': warnings,
            'is_compliant': len(violations) == 0
        }
    
    def optimize_order(self, cart_items: List[Dict], session_id: str) -> OrderOptimization:
        """Optimize order for better value and satisfaction"""
        if not self.llm or not cart_items:
            return OrderOptimization(0, 0, 0, [], "Unable to optimize order")
        
        try:
            # Calculate current total
            current_total = sum(item['price'] * item['quantity'] for item in cart_items)
            
            # Get user preferences
            user_prefs = analyze_user_preferences(session_id)
            
            # Prepare optimization prompt
            cart_summary = []
            for item in cart_items:
                cart_summary.append(f"{item['quantity']}x {item['name']} (${item['price']:.2f} each)")
            
            menu_summary = []
            for category in self.menu_data:
                for item in category.get('items', []):
                    menu_summary.append(f"{item['name']} - ${item['price']:.2f} ({category['category']})")
            
            prompt = f"""
            Analyze this food order and suggest optimizations for better value and satisfaction.
            
            Current Cart:
            {chr(10).join(cart_summary)}
            Current Total: ${current_total:.2f}
            
            User Preferences:
            - Preferred categories: {user_prefs.get('preferred_categories', [])}
            - Dietary preferences: {user_prefs.get('dietary_preferences', [])}
            - Price sensitivity: {user_prefs.get('price_sensitivity', 0.5)}
            
            Available Menu Items:
            {chr(10).join(menu_summary[:20])}
            
            Suggest optimizations considering:
            1. Better value combinations
            2. Complementary items
            3. User preferences
            4. Dietary restrictions
            5. Portion sizes
            
            Provide specific suggestions with reasoning.
            Format: suggestion|reasoning
            """
            
            response = self.llm.invoke(prompt)
            content = response.content if hasattr(response, 'content') else str(response)
            
            # Parse suggestions
            suggestions = []
            reasoning_parts = []
            
            for line in content.split('\n'):
                if '|' in line:
                    suggestion, reason = line.split('|', 1)
                    suggestions.append(suggestion.strip())
                    reasoning_parts.append(reason.strip())
                elif line.strip() and not line.startswith('#'):
                    suggestions.append(line.strip())
            
            # Estimate potential savings (simplified)
            estimated_savings = current_total * 0.05  # 5% average savings
            optimized_total = current_total - estimated_savings
            
            return OrderOptimization(
                original_total=current_total,
                optimized_total=optimized_total,
                savings=estimated_savings,
                suggestions=suggestions[:5],  # Top 5 suggestions
                reasoning='; '.join(reasoning_parts[:3])
            )
            
        except Exception as e:
            print(f"Order optimization failed: {e}")
            return OrderOptimization(current_total, current_total, 0, [], "Optimization unavailable")
    
    def suggest_complementary_items(self, cart_items: List[Dict], k: int = 3) -> List[Dict]:
        """Suggest items that complement the current cart"""
        if not self.llm or not cart_items:
            return []
        
        try:
            cart_categories = set()
            cart_items_names = []
            
            for item in cart_items:
                menu_item = self.find_menu_item(item['name'])
                if menu_item:
                    cart_categories.add(menu_item.get('category', ''))
                cart_items_names.append(item['name'])
            
            prompt = f"""
            Current order contains: {', '.join(cart_items_names)}
            Categories: {', '.join(cart_categories)}
            
            Suggest {k} complementary items that would go well with this order.
            Consider:
            1. Flavor combinations
            2. Meal balance (appetizer, main, dessert, drink)
            3. Popular pairings
            4. Nutritional balance
            
            Return only item names that exist in our menu, one per line.
            """
            
            response = self.llm.invoke(prompt)
            content = response.content if hasattr(response, 'content') else str(response)
            
            suggested_names = [line.strip() for line in content.split('\n') if line.strip()]
            
            # Find actual menu items
            suggestions = []
            for name in suggested_names[:k]:
                menu_item = self.find_menu_item(name)
                if menu_item:
                    suggestions.append(menu_item)
            
            return suggestions
            
        except Exception as e:
            print(f"Complementary item suggestion failed: {e}")
            return []

# Initialize smart order manager
smart_order_manager = SmartOrderManager()

@tool
def intelligent_menu_search(query: str, session_id: str = "", dietary_preferences: str = "") -> str:
    """
    Intelligently search the menu using advanced RAG and personalization.
    Handles natural language queries about food items, categories, dietary restrictions, etc.
    """
    try:
        # Extract session_id from query if not provided
        if not session_id:
            session_id = extract_session_id_from_input(query)
        
        # Parse dietary preferences
        dietary_prefs = [pref.strip() for pref in dietary_preferences.split(',')] if dietary_preferences else []
        
        # Use intelligent RAG for search
        result = search_menu(
            query=query,
            session_id=session_id,
            dietary_prefs=dietary_prefs
        )
        
        return result
        
    except Exception as e:
        return f"I'm having trouble searching the menu right now. Error: {str(e)}"

@tool
def smart_add_to_cart(session_id: str, items_description: str, dietary_preferences: str = "") -> str:
    """
    Intelligently add items to cart with AI-powered item recognition, context awareness, and smart defaults.
    Enhanced with better fuzzy matching, conversation context, and robust error handling.
    """
    try:
        # Extract session_id from items_description if not provided
        if not session_id:
            session_id = extract_session_id_from_input(items_description)
        
        if not session_id:
            return "⚠️ Unable to process request: Session ID not found. Please try again."
        
        # Get conversation context to apply previous preferences
        from database import get_conversations
        recent_conversations = get_conversations(session_id, limit=8)
        
        # Extract user preferences from recent conversation with enhanced parsing
        size_preference = None
        crust_preference = None
        delivery_mode = None
        spice_level = None
        
        for conv in recent_conversations:
            content = conv.get('content', '').lower()
            if conv.get('role') == 'user':
                # Extract size preferences
                if any(size in content for size in ['small', 'medium', 'large']):
                    if 'small' in content:
                        size_preference = 'small'
                    elif 'large' in content:
                        size_preference = 'large'
                    elif 'medium' in content:
                        size_preference = 'medium'
                
                # Extract crust preferences
                if any(crust in content for crust in ['thin', 'thick', 'crispy', 'regular']):
                    if 'thin' in content or 'crispy' in content:
                        crust_preference = 'thin'
                    elif 'thick' in content:
                        crust_preference = 'thick'
                    elif 'regular' in content:
                        crust_preference = 'regular'
                
                # Extract delivery preferences
                if 'delivery' in content:
                    delivery_mode = 'delivery'
                elif 'pickup' in content or 'pick up' in content:
                    delivery_mode = 'pickup'
                
                # Extract spice level
                if any(spice in content for spice in ['mild', 'medium spicy', 'spicy', 'extra spicy']):
                    if 'extra spicy' in content:
                        spice_level = 'extra spicy'
                    elif 'medium spicy' in content:
                        spice_level = 'medium spicy'  
                    elif 'spicy' in content:
                        spice_level = 'spicy'
                    elif 'mild' in content:
                        spice_level = 'mild'
        
        # Use NLU to extract items with smart defaults
        nlu_result = process_user_input(f"add {items_description}", session_id)
        
        if not nlu_result.get('menu_items'):
            # Enhanced intelligent parsing for various input formats
            import re
            
            # Handle multiple parsing patterns
            parsing_patterns = [
                r'(\d+)\s*x?\s*(.+)',  # "2 pizzas", "2x pizza"
                r'(\d+)\s+(.+)',       # "2 margherita pizza"
                r'(two|three|four|five|six|seven|eight|nine|ten)\s+(.+)',  # "two pizzas"
                r'(.+)\s*x\s*(\d+)',   # "pizza x 2"
                r'(.+)',               # Just item name
            ]
            
            # Number word mapping
            word_to_num = {
                'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
                'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10
            }
            
            quantity = 1
            item_name = items_description.strip()
            
            for pattern in parsing_patterns:
                match = re.search(pattern, items_description.strip(), re.IGNORECASE)
                if match:
                    if pattern.startswith(r'(\d+)'):
                        quantity = int(match.group(1))
                        item_name = match.group(2).strip()
                        break
                    elif 'two|three|four' in pattern:
                        quantity_word = match.group(1).lower()
                        quantity = word_to_num.get(quantity_word, 1)
                        item_name = match.group(2).strip()
                        break
                    elif 'x' in pattern and len(match.groups()) == 2:
                        item_name = match.group(1).strip()
                        quantity = int(match.group(2))
                        break
                    else:
                        item_name = match.group(1).strip()
                        break
            
            # Enhanced fuzzy matching with multiple strategies
            found_item = smart_order_manager.find_menu_item(item_name)
            
            if not found_item:
                # Try alternative matching strategies
                # 1. Remove common words and try again
                cleaned_name = re.sub(r'\b(pizza|burger|sandwich|drink|beverage)\b', '', item_name, flags=re.IGNORECASE).strip()
                if cleaned_name:
                    found_item = smart_order_manager.find_menu_item(cleaned_name)
                
                # 2. Try with common misspellings
                if not found_item:
                    common_corrections = {
                        'marghetia': 'margherita',
                        'margherita': 'margherita',
                        'pepperoni': 'pepperoni',
                        'hawaian': 'hawaiian',
                        'chiken': 'chicken',
                        'checken': 'chicken',
                        'burguer': 'burger',
                        'burgur': 'burger'
                    }
                    
                    item_lower = item_name.lower()
                    for wrong, correct in common_corrections.items():
                        if wrong in item_lower:
                            corrected_name = item_lower.replace(wrong, correct)
                            found_item = smart_order_manager.find_menu_item(corrected_name)
                            if found_item:
                                break
                
                # 3. Partial matching for complex descriptions
                if not found_item:
                    words = item_name.lower().split()
                    for category in smart_order_manager.menu_data:
                        for menu_item in category.get('items', []):
                            menu_words = menu_item.get('name', '').lower().split()
                            # Check if at least 60% of words match
                            match_count = sum(1 for word in words if any(mw.startswith(word[:3]) for mw in menu_words))
                            if match_count / len(words) >= 0.6:
                                found_item = menu_item
                                found_item['category'] = category.get('category', '')
                                break
                        if found_item:
                            break
            
            if not found_item:
                # Provide intelligent suggestions
                suggestion_prompt = f"User is looking for '{item_name}'. Based on our menu, suggest 3 similar items they might want."
                return f"I couldn't find '{item_name}' on our menu. Let me search for similar items using our intelligent menu search..."
            
            # Create standardized menu items list
            nlu_result = {
                'menu_items': [{
                    'name': found_item['name'],
                    'quantity': quantity
                }]
            }
        
        cart = get_cart(session_id)
        if cart is None:
            cart = []
        
        added_items = []
        applied_preferences = []
        total_added_value = 0
        
        # Parse dietary preferences
        dietary_prefs = [pref.strip() for pref in dietary_preferences.split(',')] if dietary_preferences else []
        
        for menu_item in nlu_result['menu_items']:
            item_name = menu_item['name']
            quantity = menu_item.get('quantity', 1)
            
            # Find menu item with enhanced matching
            found_item = smart_order_manager.find_menu_item(item_name)
            
            if not found_item:
                return f"I couldn't find '{item_name}' on our menu. Would you like me to suggest similar items?"
            
            # Apply smart defaults based on conversation context
            item_details = {'name': found_item['name']}
            
            # Apply size preference if this item supports sizes
            if size_preference and any(keyword in found_item['name'].lower() for keyword in ['pizza', 'burger', 'sandwich', 'drink', 'coffee']):
                item_details['size'] = size_preference
                applied_preferences.append(f"{size_preference} size")
            
            # Apply crust preference for pizzas
            if crust_preference and 'pizza' in found_item['name'].lower():
                item_details['crust'] = crust_preference
                applied_preferences.append(f"{crust_preference} crust")
            
            # Apply spice level if relevant
            if spice_level and any(keyword in found_item['name'].lower() for keyword in ['chicken', 'curry', 'spicy', 'sauce']):
                item_details['spice_level'] = spice_level
                applied_preferences.append(f"{spice_level}")
            
            # Validate dietary restrictions
            if dietary_prefs:
                validation = smart_order_manager.validate_dietary_restrictions([{'name': found_item['name']}], dietary_prefs)
                if not validation['is_compliant']:
                    return f"⚠️ Warning: {found_item['name']} doesn't meet your dietary preferences: {', '.join(validation['violations'])}. Would you like me to suggest alternatives?"
            
            # Add to cart with robust duplicate handling
            existing_item = None
            for cart_item in cart:
                if cart_item['name'].lower() == found_item['name'].lower():
                    # Check if customizations match
                    customizations_match = True
                    for key in ['size', 'crust', 'spice_level']:
                        if item_details.get(key) != cart_item.get(key):
                            customizations_match = False
                            break
                    
                    if customizations_match:
                        existing_item = cart_item
                        break
            
            if existing_item:
                existing_item['quantity'] += quantity
                # Update preferences if new ones were applied
                if applied_preferences:
                    existing_item.update(item_details)
            else:
                cart_item = {
                    'name': found_item['name'],
                    'price': found_item['price'],
                    'quantity': quantity
                }
                cart_item.update(item_details)
                cart.append(cart_item)
            
            # Calculate item value
            item_value = found_item['price'] * quantity
            total_added_value += item_value
            
            # Format the added item description with preferences
            item_desc = f"{quantity}x {found_item['name']}"
            if applied_preferences:
                item_desc += f" ({', '.join(applied_preferences)})"
            item_desc += f" (${found_item['price']:.2f} each = ${item_value:.2f})"
            added_items.append(item_desc)
        
        # Save cart with robust error handling and retry logic
        max_retries = 3
        save_success = False
        
        for attempt in range(max_retries):
            try:
                save_result = save_cart(session_id, cart)
                if save_result:
                    save_success = True
                    break
            except Exception as e:
                if attempt == max_retries - 1:
                    return f"⚠️ I had trouble saving your cart after {max_retries} attempts: {str(e)}. Please try again."
                continue
        
        if not save_success:
            return "⚠️ I had trouble saving your cart. Please try again in a moment."
        
        # Calculate new total
        total = sum(item['price'] * item['quantity'] for item in cart)
        item_count = sum(item['quantity'] for item in cart)
        
        # Create comprehensive response
        response = f"✅ **Added to cart:** {'; '.join(added_items)}"
        
        if applied_preferences:
            unique_prefs = list(set(applied_preferences))
            response += f"\n🎯 **Applied your preferences:** {', '.join(unique_prefs)}"
        
        response += f"\n💰 **Cart total:** ${total:.2f} ({item_count} items)"
        
        # Add delivery mode if detected
        if delivery_mode:
            response += f"\n🚚 **For {delivery_mode}**"
            if delivery_mode == 'delivery':
                response += " - I'll need your address when you're ready to order"
        
        # Add helpful suggestions
        if total > 15:
            response += f"\n💡 **Tip:** Your order qualifies for free delivery! Ready to checkout?"
        elif total > 30:
            response += f"\n🎉 **Great choice!** Want me to suggest a drink or dessert to complete your meal?"
        
        # Offer order optimization for larger orders
        if len(cart) >= 3 or total > 20:
            response += f"\n🔍 **Pro tip:** I can analyze your order for potential savings. Just say 'optimize my order'!"
        
        return response
        
    except Exception as e:
        return f"I had trouble adding items to your cart: {str(e)}. Please try again or rephrase your request."

@tool
def smart_remove_from_cart(session_id: str, items_description: str) -> str:
    """
    Intelligently remove items from cart with natural language understanding.
    """
    try:
        cart = get_cart(session_id)
        
        if not cart:
            return "Your cart is empty."
        
        # Use NLU to extract items to remove
        nlu_result = process_user_input(f"remove {items_description}", session_id)
        
        if not nlu_result.get('menu_items'):
            return f"I couldn't identify which items to remove from '{items_description}'. Please specify the item name and quantity, like 'remove 1 pizza' or 'remove chicken burger'."
        
        removed_items = []
        
        for menu_item in nlu_result['menu_items']:
            item_name = menu_item['name']
            quantity_to_remove = menu_item.get('quantity', 1)
            
            # Find item in cart
            cart_item = next((item for item in cart if item_name.lower() in item['name'].lower()), None)
            
            if not cart_item:
                removed_items.append(f"{item_name} was not in your cart")
                continue
            
            # Remove quantity
            if cart_item['quantity'] <= quantity_to_remove:
                # Remove entire item
                cart = [item for item in cart if item['name'] != cart_item['name']]
                removed_items.append(f"Removed all {cart_item['name']} from cart")
            else:
                # Reduce quantity
                cart_item['quantity'] -= quantity_to_remove
                removed_items.append(f"Removed {quantity_to_remove}x {cart_item['name']}")
        
        save_cart(session_id, cart)
        
        # Calculate new total
        total = sum(item['price'] * item['quantity'] for item in cart)
        
        return f"{'; '.join(removed_items)}\nCart total: ${total:.2f}"
        
    except Exception as e:
        return f"I had trouble removing items from your cart. Error: {str(e)}"

@tool
def optimize_current_order(session_id: str) -> str:
    """
    Analyze and optimize the current cart for better value and satisfaction.
    """
    try:
        cart = get_cart(session_id)
        
        if not cart:
            return "Your cart is empty. Add some items first, and I'll help optimize your order!"
        
        optimization = smart_order_manager.optimize_order(cart, session_id)
        
        if not optimization.suggestions:
            return f"Your current order looks great! Total: ${optimization.original_total:.2f}"
        
        response = f"Order Optimization Analysis:\n"
        response += f"Current total: ${optimization.original_total:.2f}\n"
        
        if optimization.savings > 0:
            response += f"Potential savings: ${optimization.savings:.2f}\n"
        
        response += f"\nSuggestions:\n"
        for i, suggestion in enumerate(optimization.suggestions, 1):
            response += f"{i}. {suggestion}\n"
        
        if optimization.reasoning:
            response += f"\nReasoning: {optimization.reasoning}"
        
        return response
        
    except Exception as e:
        return f"I couldn't optimize your order right now. Error: {str(e)}"

@tool
def smart_order_confirmation(session_id: str, customer_details: str = "", delivery_mode: str = "", payment_method: str = "") -> str:
    """
    Intelligently confirm order with validation, optimization suggestions, and smart processing.
    Enhanced to handle delivery, payment, and customer information seamlessly.
    """
    try:
        cart = get_cart(session_id)
        
        if not cart:
            return "Your cart is empty. Please add items before confirming your order."
        
        # Parse customer details if provided
        customer_info = {}
        if customer_details:
            # Enhanced parsing for better extraction
            details_lower = customer_details.lower()
            
            # Extract phone number (various formats)
            phone_patterns = [
                r'\b(\d{10})\b',  # 10 digits
                r'\b(\d{3}[-.\s]?\d{3}[-.\s]?\d{4})\b',  # with separators
                r'\+1[-.\s]?(\d{3}[-.\s]?\d{3}[-.\s]?\d{4})\b'  # with country code
            ]
            for pattern in phone_patterns:
                phone_match = re.search(pattern, customer_details)
                if phone_match:
                    # Clean phone number (remove separators)
                    phone = re.sub(r'[^\d]', '', phone_match.group(1) if phone_match.group(1) else phone_match.group(0))
                    if len(phone) == 10:
                        customer_info['phone'] = phone
                        break
            
            # Extract name with improved patterns
            name_patterns = [
                r'name[:\s]+([a-zA-Z\s]+?)(?:\s|$|,|\.|phone|address)',
                r'i\'?m\s+([a-zA-Z\s]+?)(?:\s|$|,|\.|phone|address)',
                r'this is\s+([a-zA-Z\s]+?)(?:\s|$|,|\.|phone|address)',
                r'my name is\s+([a-zA-Z\s]+?)(?:\s|$|,|\.|phone|address)',
                r'^([a-zA-Z\s]{2,30})(?:\s|$|,|\.|phone|address)'  # First words if looks like name
            ]
            for pattern in name_patterns:
                match = re.search(pattern, customer_details, re.IGNORECASE)
                if match:
                    name = match.group(1).strip().title()
                    # Validate name (not just numbers or single letter)
                    if len(name) > 1 and not name.isdigit() and ' ' in name or len(name.split()) == 1:
                        customer_info['name'] = name
                        break
            
            # Extract address
            address_patterns = [
                r'address[:\s]+(.+?)(?:\s|$|phone|name)',
                r'deliver to\s+(.+?)(?:\s|$|phone|name)',
                r'address is\s+(.+?)(?:\s|$|phone|name)'
            ]
            for pattern in address_patterns:
                match = re.search(pattern, customer_details, re.IGNORECASE)
                if match:
                    address = match.group(1).strip()
                    if len(address) > 5:  # Reasonable address length
                        customer_info['address'] = address
                        break
        
        # Get conversation context for missing information
        from database import get_conversations
        recent_conversations = get_conversations(session_id, limit=8)
        
        # Extract information from conversation history if not provided
        for conv in recent_conversations:
            content = conv.get('content', '').lower()
            role = conv.get('role', '')
            
            if role == 'user':
                # Extract delivery mode from conversation
                if not delivery_mode:
                    if 'delivery' in content:
                        delivery_mode = 'delivery'
                    elif 'pickup' in content or 'pick up' in content:
                        delivery_mode = 'pickup'
                
                # Extract payment method from conversation
                if not payment_method:
                    if 'cash' in content:
                        payment_method = 'cash'
                    elif 'card' in content or 'credit' in content or 'debit' in content:
                        payment_method = 'card'
                    elif 'upi' in content or 'gpay' in content or 'paytm' in content:
                        payment_method = 'upi'
                    elif 'online' in content:
                        payment_method = 'online'
                
                # Extract customer info from conversation if not already found
                if not customer_info.get('name'):
                    name_match = re.search(r'my name is\s+([a-zA-Z\s]+)', content)
                    if name_match:
                        customer_info['name'] = name_match.group(1).strip().title()
                
                if not customer_info.get('phone'):
                    phone_match = re.search(r'\b(\d{10})\b', content)
                    if phone_match:
                        customer_info['phone'] = phone_match.group(1)
                
                if not customer_info.get('address'):
                    # Look for address patterns in conversation
                    address_indicators = ['address', 'street', 'apartment', 'house', 'building']
                    for indicator in address_indicators:
                        if indicator in content:
                            # Extract potential address
                            words = content.split()
                            try:
                                idx = words.index(indicator)
                                if idx < len(words) - 1:
                                    # Take next few words as address
                                    address_candidate = ' '.join(words[idx+1:idx+6])
                                    if len(address_candidate.strip()) > 5:
                                        customer_info['address'] = address_candidate.strip()
                                        break
                            except ValueError:
                                continue
        
        # Validate required information with intelligent defaults
        missing_info = []
        if not customer_info.get('name'):
            missing_info.append('your name')
        if not customer_info.get('phone'):
            missing_info.append('phone number')
        
        # Delivery mode validation
        if not delivery_mode:
            delivery_mode = 'delivery'  # Default to delivery
        
        # Address validation for delivery
        if delivery_mode == 'delivery' and not customer_info.get('address'):
            missing_info.append('delivery address')
        
        # Payment method default
        if not payment_method:
            payment_method = 'cash'  # Default to cash on delivery
        
        if missing_info:
            missing_str = ', '.join(missing_info[:-1]) + (' and ' + missing_info[-1] if len(missing_info) > 1 else missing_info[0])
            return f"To confirm your order, I need {missing_str}. Please provide this information or update your profile in the sidebar.\n\nExample: 'My name is John Doe, phone 9876543210, address 123 Main Street'"
        
        # Enhanced phone validation
        phone = customer_info.get('phone', '')
        if not (phone.isdigit() and len(phone) == 10):
            return "Please provide a valid 10-digit phone number (e.g., 9876543210)."
        
        # Name validation
        name = customer_info.get('name', '')
        if len(name.strip()) < 2:
            return "Please provide your full name."
        
        # Calculate costs with comprehensive breakdown
        subtotal = sum(item['price'] * item['quantity'] for item in cart)
        
        # Delivery charges
        delivery_charge = 0
        if delivery_mode == 'delivery':
            if subtotal < 20:  # Minimum order for free delivery
                delivery_charge = 2.99
        
        # Payment processing fee (for online payments)
        processing_fee = 0
        if payment_method in ['card', 'upi', 'online']:
            processing_fee = subtotal * 0.02  # 2% processing fee
        
        # Tax calculation
        tax = (subtotal + delivery_charge) * 0.08  # 8% tax
        
        # Final total
        total = subtotal + delivery_charge + processing_fee + tax
        
        # Final optimization check with suggestions
        optimization = smart_order_manager.optimize_order(cart, session_id)
        optimization_savings = ""
        if optimization.suggestions and optimization.savings > 1:
            optimization_savings = f"\n💡 Tip: {optimization.suggestions[0]} (Could save ~${optimization.savings:.2f})"
        
        # Dietary validation
        dietary_prefs = analyze_user_preferences(session_id).get('dietary_preferences', [])
        dietary_warnings = ""
        if dietary_prefs:
            validation = smart_order_manager.validate_dietary_restrictions(cart, dietary_prefs)
            if validation['violations']:
                dietary_warnings = f"\n⚠️ Dietary Alert: {'; '.join(validation['violations'])}"
        
        # Create comprehensive order
        order_details = {
            'name': customer_info['name'],
            'phone': customer_info['phone'],
            'address': customer_info.get('address', 'Pickup from restaurant'),
            'delivery_mode': delivery_mode,
            'payment_method': payment_method,
            'special_instructions': '',
            'dietary_preferences': dietary_prefs
        }
        
        # Insert order with enhanced details
        order_id = insert_order(session_id, cart, total, order_details)
        
        # Clear cart after successful order
        save_cart(session_id, [])
        
        # Generate comprehensive confirmation message
        items_summary = []
        for item in cart:
            item_line = f"{item['quantity']}x {item['name']}"
            # Add customizations if available
            if item.get('size'):
                item_line += f" ({item['size']} size)"
            if item.get('crust'):
                item_line += f" ({item['crust']} crust)"
            item_line += f" - ${item['price'] * item['quantity']:.2f}"
            items_summary.append(item_line)
        
        # Estimated delivery time
        if delivery_mode == 'delivery':
            delivery_time = "25-35 minutes"
            delivery_info = f"\n🚚 Delivery to: {customer_info['address']}"
        else:
            delivery_time = "15-20 minutes"
            delivery_info = f"\n🏪 Pickup from: Restaurant Location"
        
        # Payment info
        payment_info = {
            'cash': 'Cash on Delivery',
            'card': 'Card Payment (Online)',
            'upi': 'UPI Payment',
            'online': 'Online Payment'
        }.get(payment_method, payment_method.title())
        
        confirmation = f"""
🎉 **ORDER CONFIRMED!** 
Order #{order_id}

👤 **Customer Details:**
• Name: {customer_info['name']}
• Phone: {customer_info['phone']}

🍽️ **Your Order:**
{chr(10).join(f"• {item}" for item in items_summary)}

💰 **Order Summary:**
• Subtotal: ${subtotal:.2f}
{f"• Delivery: ${delivery_charge:.2f}" if delivery_charge > 0 else "• Delivery: FREE"}
{f"• Processing Fee: ${processing_fee:.2f}" if processing_fee > 0 else ""}
• Tax (8%): ${tax:.2f}
• **Total: ${total:.2f}**

{delivery_info}
⏰ Estimated {delivery_mode}: {delivery_time}
💳 Payment: {payment_info}

Thank you for your order! We'll start preparing it right away.
{optimization_savings}
{dietary_warnings}
        """.strip()
        
        return confirmation
        
    except Exception as e:
        return f"I encountered an error while processing your order: {str(e)}. Please try again or contact support."

@tool
def get_smart_recommendations(session_id: str, context: str = "", preferences: str = "") -> str:
    """
    Get intelligent, personalized recommendations based on user history and context.
    """
    try:
        # Parse preferences
        dietary_prefs = [pref.strip() for pref in preferences.split(',')] if preferences else []
        
        # Get personalized recommendations
        recommendations = get_personalized_recommendations(
            session_id=session_id,
            context=context,
            dietary_prefs=dietary_prefs,
            k=5
        )
        
        return recommendations
        
    except Exception as e:
        return f"I'd love to recommend some items! Try our popular Margherita Pizza, Chicken Burger, or Fresh Lemonade. Error: {str(e)}"

@tool
def analyze_nutritional_info(session_id: str, items_query: str = "") -> str:
    """
    Provide detailed nutritional analysis for menu items or current cart.
    """
    try:
        if items_query:
            # Analyze specific items
            nlu_result = process_user_input(f"nutrition info for {items_query}", session_id)
            items_to_analyze = []
            
            if nlu_result.get('menu_items'):
                for menu_item in nlu_result['menu_items']:
                    found_item = smart_order_manager.find_menu_item(menu_item['name'])
                    if found_item:
                        items_to_analyze.append(found_item)
            
            if not items_to_analyze:
                return f"I couldn't find nutritional information for '{items_query}'. Please specify menu item names."
        else:
            # Analyze current cart
            cart = get_cart(session_id)
            if not cart:
                return "Your cart is empty. Add items to see nutritional information."
            
            items_to_analyze = []
            for cart_item in cart:
                menu_item = smart_order_manager.find_menu_item(cart_item['name'])
                if menu_item:
                    # Multiply by quantity
                    item_copy = menu_item.copy()
                    nutrition = item_copy.get('nutrition', {})
                    quantity = cart_item['quantity']
                    
                    # Scale nutrition by quantity
                    scaled_nutrition = {}
                    for key, value in nutrition.items():
                        if isinstance(value, (int, float)):
                            scaled_nutrition[key] = value * quantity
                        else:
                            scaled_nutrition[key] = value
                    
                    item_copy['nutrition'] = scaled_nutrition
                    item_copy['quantity'] = quantity
                    items_to_analyze.append(item_copy)
        
        if not items_to_analyze:
            return "No nutritional information available for the requested items."
        
        # Generate nutritional analysis
        analysis = "Nutritional Information:\n\n"
        
        total_calories = 0
        total_protein = 0
        total_carbs = 0
        total_fat = 0
        
        for item in items_to_analyze:
            nutrition = item.get('nutrition', {})
            quantity = item.get('quantity', 1)
            
            analysis += f"• {item['name']}"
            if quantity > 1:
                analysis += f" (x{quantity})"
            analysis += f":\n"
            
            calories = nutrition.get('calories', 0)
            protein = nutrition.get('protein', 0)
            carbs = nutrition.get('carbs', 0)
            fat = nutrition.get('fat', 0)
            
            analysis += f"  Calories: {calories}\n"
            analysis += f"  Protein: {protein}g\n"
            analysis += f"  Carbs: {carbs}g\n"
            analysis += f"  Fat: {fat}g\n"
            
            if nutrition.get('fiber'):
                analysis += f"  Fiber: {nutrition['fiber']}g\n"
            if nutrition.get('sodium'):
                analysis += f"  Sodium: {nutrition['sodium']}mg\n"
            
            # Add dietary tags
            dietary_tags = item.get('dietary_tags', [])
            if dietary_tags:
                analysis += f"  Dietary: {', '.join(dietary_tags)}\n"
            
            analysis += "\n"
            
            # Add to totals
            total_calories += calories
            total_protein += protein
            total_carbs += carbs
            total_fat += fat
        
        # Add totals if multiple items
        if len(items_to_analyze) > 1:
            analysis += f"TOTALS:\n"
            analysis += f"Total Calories: {total_calories}\n"
            analysis += f"Total Protein: {total_protein}g\n"
            analysis += f"Total Carbs: {total_carbs}g\n"
            analysis += f"Total Fat: {total_fat}g\n"
        
        return analysis
        
    except Exception as e:
        return f"I couldn't analyze nutritional information right now. Error: {str(e)}"

@tool
def smart_order_tracking(session_id: str, order_query: str = "") -> str:
    """
    Intelligently track orders and provide status updates.
    """
    try:
        orders = get_order_history(session_id)
        
        if not orders:
            return "You don't have any orders yet. Place your first order to start tracking!"
        
        if order_query:
            # Try to extract order ID
            order_id_match = re.search(r'\b(\d+)\b', order_query)
            if order_id_match:
                order_id = int(order_id_match.group(1))
                specific_order = next((order for order in orders if order['id'] == order_id), None)
                
                if specific_order:
                    items_summary = ', '.join([f"{item['quantity']}x {item['name']}" for item in specific_order['items']])
                    return f"Order #{order_id}:\nStatus: {specific_order['status']}\nItems: {items_summary}\nTotal: ${specific_order['total']:.2f}\nDate: {specific_order['time']}"
                else:
                    return f"I couldn't find order #{order_id}. Your recent orders are: {', '.join([str(o['id']) for o in orders[:5]])}"
        
        # Show recent orders
        recent_orders = orders[:3]  # Last 3 orders
        
        response = "Your Recent Orders:\n\n"
        for order in recent_orders:
            items_summary = ', '.join([f"{item['quantity']}x {item['name']}" for item in order['items']])
            response += f"Order #{order['id']} - {order['status']}\n"
            response += f"Items: {items_summary}\n"
            response += f"Total: ${order['total']:.2f} | Date: {order['time']}\n\n"
        
        # Add pending order info
        pending_orders = [o for o in orders if o['status'] == 'Pending']
        if pending_orders:
            response += f"You have {len(pending_orders)} pending order(s) that can be cancelled if needed."
        
        return response
        
    except Exception as e:
        return f"I couldn't retrieve your order information. Error: {str(e)}"

@tool
def complete_order_flow(session_id: str, user_request: str = "") -> str:
    """
    Complete end-to-end order flow with intelligent guidance.
    Handles cart review, customer details collection, and order confirmation.
    """
    try:
        cart = get_cart(session_id)
        
        if not cart:
            return """
🛒 **Your cart is empty!** Let's start your order:

**Popular choices to get you started:**
• 🍕 "Add 2 Margherita pizzas"
• 🍔 "I want a chicken burger and fries"
• 🥤 "Add a large coke"

**Or browse our menu:**
• "Show me pizza options"
• "What desserts do you have?"
• "Give me healthy recommendations"

Just tell me what you'd like to eat! 😊
            """.strip()
        
        # Analyze current cart status
        total = sum(item['price'] * item['quantity'] for item in cart)
        item_count = sum(item['quantity'] for item in cart)
        
        # Check what information we're missing
        from database import get_conversations
        conversations = get_conversations(session_id, limit=10)
        
        # Extract customer info from conversations
        customer_info = {}
        delivery_mode = None
        payment_method = None
        
        for conv in conversations:
            content = conv.get('content', '').lower()
            if conv.get('role') == 'user':
                # Look for customer details
                name_match = re.search(r'(?:name|i\'m|this is)\s+([a-zA-Z\s]{2,30})', content)
                if name_match and not customer_info.get('name'):
                    customer_info['name'] = name_match.group(1).strip().title()
                
                phone_match = re.search(r'\b(\d{10})\b', content)
                if phone_match and not customer_info.get('phone'):
                    customer_info['phone'] = phone_match.group(1)
                
                if 'delivery' in content and not delivery_mode:
                    delivery_mode = 'delivery'
                elif ('pickup' in content or 'pick up' in content) and not delivery_mode:
                    delivery_mode = 'pickup'
                
                if any(payment in content for payment in ['cash', 'card', 'upi']) and not payment_method:
                    if 'cash' in content:
                        payment_method = 'cash'
                    elif 'card' in content:
                        payment_method = 'card'
                    elif 'upi' in content:
                        payment_method = 'upi'
        
        # Build order summary
        items_summary = []
        for item in cart:
            item_desc = f"• {item['quantity']}x {item['name']}"
            if item.get('size'):
                item_desc += f" ({item['size']})"
            if item.get('crust'):
                item_desc += f" - {item['crust']} crust"
            item_desc += f" - ${item['price'] * item['quantity']:.2f}"
            items_summary.append(item_desc)
        
        response = f"""
🛒 **ORDER SUMMARY** ({item_count} items)
{chr(10).join(items_summary)}

💰 **Total: ${total:.2f}**
        """.strip()
        
        # Check what's missing and guide user
        missing_items = []
        if not customer_info.get('name'):
            missing_items.append("your name")
        if not customer_info.get('phone'):
            missing_items.append("phone number")
        if not delivery_mode:
            missing_items.append("delivery preference (delivery/pickup)")
        if delivery_mode == 'delivery' and not customer_info.get('address'):
            missing_items.append("delivery address")
        
        if missing_items:
            response += f"\n\n📝 **To complete your order, I need:**"
            for i, item in enumerate(missing_items, 1):
                response += f"\n{i}. {item.title()}"
            
            response += "\n\n**Quick format:** \"My name is John Doe, phone 9876543210, delivery to 123 Main Street\""
            response += "\n**Or say:** \"Pickup, name John, phone 9876543210\""
        else:
            # Ready to confirm
            delivery_info = f"{delivery_mode}"
            if delivery_mode == 'delivery':
                delivery_info += f" to {customer_info.get('address', 'provided address')}"
            
            response += f"""

✅ **READY TO ORDER!**
👤 **Customer:** {customer_info['name']}
📞 **Phone:** {customer_info['phone']}
🚚 **Service:** {delivery_info}
💳 **Payment:** {payment_method or 'Cash on delivery'}

**Say "confirm my order" to place it!**
            """.strip()
        
        # Add helpful suggestions
        if total < 15:
            response += f"\n\n💡 **Tip:** Add ${15 - total:.2f} more for free delivery!"
        
        response += f"\n\n**Need changes?**"
        response += f"\n• \"Add more items\" - Browse menu"
        response += f"\n• \"Remove [item]\" - Modify cart"
        response += f"\n• \"Optimize my order\" - Get savings tips"
        
        return response
        
    except Exception as e:
        return f"I had trouble processing your order flow: {str(e)}. Let me help you step by step."

# Export all tools for easy import
SMART_TOOLS = [
    intelligent_menu_search,
    smart_add_to_cart,
    smart_remove_from_cart,
    optimize_current_order,
    smart_order_confirmation,
    get_smart_recommendations,
    analyze_nutritional_info,
    smart_order_tracking,
    complete_order_flow
]