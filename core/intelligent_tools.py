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
                model="llama3-70b-8192",
                groq_api_key=GROQ_API_KEY,
                temperature=0.2,
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
    Intelligently add items to cart with AI-powered item recognition and validation.
    Handles natural language descriptions and suggests alternatives if needed.
    """
    try:
        # Use NLU to extract items
        nlu_result = process_user_input(f"add {items_description}", session_id)
        
        if not nlu_result.get('menu_items'):
            return f"I couldn't identify specific menu items from '{items_description}'. Could you be more specific? For example: 'add 2 margherita pizzas' or 'add chicken burger'."
        
        cart = get_cart(session_id)
        added_items = []
        
        # Parse dietary preferences
        dietary_prefs = [pref.strip() for pref in dietary_preferences.split(',')] if dietary_preferences else []
        
        for menu_item in nlu_result['menu_items']:
            item_name = menu_item['name']
            quantity = menu_item.get('quantity', 1)
            
            # Find menu item
            found_item = smart_order_manager.find_menu_item(item_name)
            
            if not found_item:
                return f"I couldn't find '{item_name}' on our menu. Would you like me to suggest similar items?"
            
            # Validate dietary restrictions
            if dietary_prefs:
                validation = smart_order_manager.validate_dietary_restrictions([{'name': found_item['name']}], dietary_prefs)
                if not validation['is_compliant']:
                    return f"Warning: {found_item['name']} doesn't meet your dietary preferences: {', '.join(validation['violations'])}. Would you like me to suggest alternatives?"
            
            # Add to cart
            existing_item = next((item for item in cart if item['name'].lower() == found_item['name'].lower()), None)
            
            if existing_item:
                existing_item['quantity'] += quantity
            else:
                cart.append({
                    'name': found_item['name'],
                    'price': found_item['price'],
                    'quantity': quantity
                })
            
            added_items.append(f"{quantity}x {found_item['name']} (${found_item['price']:.2f} each)")
        
        save_cart(session_id, cart)
        
        # Calculate new total
        total = sum(item['price'] * item['quantity'] for item in cart)
        
        # Suggest complementary items
        suggestions = smart_order_manager.suggest_complementary_items(cart, k=2)
        suggestion_text = ""
        if suggestions:
            suggestion_names = [item['name'] for item in suggestions]
            suggestion_text = f"\n\nYou might also like: {', '.join(suggestion_names)}"
        
        return f"Added to cart: {'; '.join(added_items)}\nCart total: ${total:.2f}{suggestion_text}"
        
    except Exception as e:
        return f"I had trouble adding items to your cart. Error: {str(e)}"

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
def smart_order_confirmation(session_id: str, customer_details: str = "") -> str:
    """
    Intelligently confirm order with validation, optimization suggestions, and smart processing.
    """
    try:
        cart = get_cart(session_id)
        
        if not cart:
            return "Your cart is empty. Please add items before confirming your order."
        
        # Parse customer details if provided
        customer_info = {}
        if customer_details:
            # Simple parsing - in production, use more sophisticated NLP
            details_lower = customer_details.lower()
            
            # Extract phone number
            phone_match = re.search(r'\b\d{10}\b', customer_details)
            if phone_match:
                customer_info['phone'] = phone_match.group()
            
            # Extract name (simple heuristic)
            name_patterns = [
                r'name[:\s]+([a-zA-Z\s]+)',
                r'i\'?m\s+([a-zA-Z\s]+)',
                r'this is\s+([a-zA-Z\s]+)'
            ]
            for pattern in name_patterns:
                match = re.search(pattern, customer_details, re.IGNORECASE)
                if match:
                    customer_info['name'] = match.group(1).strip()
                    break
        
        # Validate required information
        missing_info = []
        if not customer_info.get('name'):
            missing_info.append('name')
        if not customer_info.get('phone'):
            missing_info.append('phone number')
        
        if missing_info:
            return f"I need your {' and '.join(missing_info)} to confirm the order. Please provide these details or save them in the sidebar."
        
        # Validate phone number
        phone = customer_info.get('phone', '')
        if not (phone.isdigit() and len(phone) == 10):
            return "Please provide a valid 10-digit phone number."
        
        # Calculate totals
        subtotal = sum(item['price'] * item['quantity'] for item in cart)
        tax = subtotal * 0.08  # 8% tax
        total = subtotal + tax
        
        # Final optimization check
        optimization = smart_order_manager.optimize_order(cart, session_id)
        optimization_note = ""
        if optimization.suggestions:
            optimization_note = f"\n\nOptimization tip: {optimization.suggestions[0]}"
        
        # Create order
        order_details = {
            'name': customer_info['name'],
            'phone': customer_info['phone'],
            'address': customer_info.get('address', 'Not provided')
        }
        
        insert_order(session_id, cart, total, order_details)
        save_cart(session_id, [])  # Clear cart
        
        # Generate confirmation message
        items_summary = []
        for item in cart:
            items_summary.append(f"{item['quantity']}x {item['name']} (${item['price'] * item['quantity']:.2f})")
        
        confirmation = f"Order confirmed! 🎉\n\n"
        confirmation += f"Customer: {customer_info['name']}\n"
        confirmation += f"Phone: {customer_info['phone']}\n\n"
        confirmation += f"Items:\n" + "\n".join(f"• {item}" for item in items_summary)
        confirmation += f"\n\nSubtotal: ${subtotal:.2f}"
        confirmation += f"\nTax: ${tax:.2f}"
        confirmation += f"\nTotal: ${total:.2f}"
        confirmation += f"\n\nEstimated delivery: 25-35 minutes"
        confirmation += optimization_note
        
        return confirmation
        
    except Exception as e:
        return f"I couldn't confirm your order. Error: {str(e)}"

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

# Export all tools for easy import
SMART_TOOLS = [
    intelligent_menu_search,
    smart_add_to_cart,
    smart_remove_from_cart,
    optimize_current_order,
    smart_order_confirmation,
    get_smart_recommendations,
    analyze_nutritional_info,
    smart_order_tracking
]