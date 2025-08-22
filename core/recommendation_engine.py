import json
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass
from collections import defaultdict, Counter
import numpy as np

from config import get_config, GROQ_API_KEY
from database import get_order_history, log_conversation
from langchain_groq import ChatGroq

@dataclass
class UserPreference:
    category: str
    item_name: str
    preference_score: float
    frequency: int
    last_ordered: datetime
    dietary_tags: List[str]
    price_sensitivity: float = 0.5 

@dataclass
class RecommendationItem:
    name: str
    category: str
    price: float
    confidence_score: float
    reason: str
    dietary_tags: List[str]
    nutrition_info: Dict[str, Any]
    estimated_preference: float

class PersonalizationEngine:
    """Advanced personalization and recommendation system"""
    
    def __init__(self):
        self.config = get_config("personalization")
        self.llm = self._create_llm()
        self.menu_data = self._load_menu_data()
        self.user_profiles = {}  # Cache for user profiles
        
    def _create_llm(self) -> Optional[ChatGroq]:
        try:
            return ChatGroq(
                model="llama-3.3-70b-versatile",
                groq_api_key=GROQ_API_KEY,
                temperature=0.3,
                max_tokens=1024
            )
        except Exception as e:
            print(f"Failed to create LLM for personalization: {e}")
            return None
    
    def _load_menu_data(self) -> List[Dict]:
        """Load enhanced menu data"""
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
        except Exception as e:
            print(f"Failed to load menu data: {e}")
            return []
    
    def extract_user_preferences(self, session_id: str) -> List[UserPreference]:
        """Extract user preferences from order history"""
        preferences = []
        
        try:
            # Get order history
            orders = get_order_history(session_id)
            
            if not orders:
                return preferences
            
            # Analyze order patterns
            item_stats = defaultdict(lambda: {
                'count': 0, 
                'total_spent': 0, 
                'last_ordered': None,
                'categories': set(),
                'dietary_tags': set()
            })
            
            for order in orders:
                order_date = datetime.fromisoformat(order['time']) if isinstance(order['time'], str) else order['time']
                
                for item in order['items']:
                    item_name = item['name']
                    item_stats[item_name]['count'] += item['quantity']
                    item_stats[item_name]['total_spent'] += item['price'] * item['quantity']
                    
                    if not item_stats[item_name]['last_ordered'] or order_date > item_stats[item_name]['last_ordered']:
                        item_stats[item_name]['last_ordered'] = order_date
                    
                    # Find menu item details
                    menu_item = self._find_menu_item(item_name)
                    if menu_item:
                        item_stats[item_name]['categories'].add(menu_item.get('category', ''))
                        item_stats[item_name]['dietary_tags'].update(menu_item.get('dietary_tags', []))
            
            # Convert to preferences
            total_orders = len(orders)
            for item_name, stats in item_stats.items():
                # Calculate preference score based on frequency and recency
                frequency_score = stats['count'] / total_orders
                
                # Recency bonus (more recent orders get higher scores)
                days_since_last = (datetime.now() - stats['last_ordered']).days
                recency_score = max(0, 1 - (days_since_last / 30))  # Decay over 30 days
                
                preference_score = (frequency_score * 0.7) + (recency_score * 0.3)
                
                # Price sensitivity (lower for expensive items ordered frequently)
                avg_price = stats['total_spent'] / stats['count']
                price_sensitivity = max(0, 1 - (avg_price / 20))  # Normalize around $20
                
                preference = UserPreference(
                    category=list(stats['categories'])[0] if stats['categories'] else 'Unknown',
                    item_name=item_name,
                    preference_score=preference_score,
                    frequency=stats['count'],
                    last_ordered=stats['last_ordered'],
                    dietary_tags=list(stats['dietary_tags']),
                    price_sensitivity=price_sensitivity
                )
                preferences.append(preference)
            
            # Sort by preference score
            preferences.sort(key=lambda x: x.preference_score, reverse=True)
            
        except Exception as e:
            print(f"Error extracting user preferences: {e}")
        
        return preferences
    
    def _find_menu_item(self, item_name: str) -> Optional[Dict]:
        """Find menu item by name"""
        item_name_lower = item_name.lower()
        
        for category in self.menu_data:
            for item in category.get('items', []):
                if item.get('name', '').lower() == item_name_lower:
                    item['category'] = category.get('category', '')
                    return item
        return None
    
    def analyze_dietary_preferences(self, session_id: str) -> List[str]:
        """Analyze user's dietary preferences from order history"""
        preferences = self.extract_user_preferences(session_id)
        
        # Count dietary tags from ordered items
        dietary_counter = Counter()
        for pref in preferences:
            for tag in pref.dietary_tags:
                dietary_counter[tag] += pref.frequency
        
        # Return most common dietary preferences
        return [tag for tag, count in dietary_counter.most_common(5)]
    
    def get_price_sensitivity(self, session_id: str) -> float:
        """Calculate user's price sensitivity"""
        orders = get_order_history(session_id)
        
        if not orders:
            return 0.5  # Default moderate sensitivity
        
        # Calculate average order value and price distribution
        order_values = [order['total'] for order in orders]
        avg_order_value = np.mean(order_values)
        
        # Users with higher average order values are less price sensitive
        # Normalize around $25 average order
        price_sensitivity = max(0, min(1, 1 - (avg_order_value / 50)))
        
        return price_sensitivity
    
    def collaborative_filtering_recommendations(self, session_id: str, k: int = 5) -> List[RecommendationItem]:
        """Generate recommendations using collaborative filtering"""
        recommendations = []
        
        try:
            # This is a simplified version - in production, you'd use a proper CF algorithm
            user_preferences = self.extract_user_preferences(session_id)
            
            if not user_preferences:
                return self.get_popular_items(k)
            
            # Find similar categories and items
            preferred_categories = set(pref.category for pref in user_preferences[:3])
            preferred_dietary_tags = set()
            for pref in user_preferences:
                preferred_dietary_tags.update(pref.dietary_tags)
            
            # Score all menu items
            item_scores = []
            for category in self.menu_data:
                category_name = category.get('category', '')
                
                for item in category.get('items', []):
                    score = 0
                    
                    # Category preference bonus
                    if category_name in preferred_categories:
                        score += 0.4
                    
                    # Dietary preference bonus
                    item_dietary_tags = set(item.get('dietary_tags', []))
                    dietary_overlap = len(preferred_dietary_tags.intersection(item_dietary_tags))
                    score += dietary_overlap * 0.2
                    
                    # Popularity bonus
                    score += item.get('popularity_score', 0.5) * 0.3
                    
                    # Price consideration
                    price_sensitivity = self.get_price_sensitivity(session_id)
                    if item.get('price', 0) < 15:  # Affordable items
                        score += price_sensitivity * 0.1
                    
                    if score > 0.3:  # Minimum threshold
                        recommendation = RecommendationItem(
                            name=item.get('name', ''),
                            category=category_name,
                            price=item.get('price', 0),
                            confidence_score=score,
                            reason=f"Based on your preference for {category_name} items",
                            dietary_tags=item.get('dietary_tags', []),
                            nutrition_info=item.get('nutrition', {}),
                            estimated_preference=score
                        )
                        item_scores.append(recommendation)
            
            # Sort by score and return top k
            item_scores.sort(key=lambda x: x.confidence_score, reverse=True)
            recommendations = item_scores[:k]
            
        except Exception as e:
            print(f"Collaborative filtering error: {e}")
            recommendations = self.get_popular_items(k)
        
        return recommendations
    
    def content_based_recommendations(self, session_id: str, k: int = 5) -> List[RecommendationItem]:
        """Generate recommendations using content-based filtering"""
        recommendations = []
        
        try:
            user_preferences = self.extract_user_preferences(session_id)
            
            if not user_preferences:
                return self.get_popular_items(k)
            
            # Create user profile vector
            user_dietary_prefs = set()
            user_categories = set()
            user_ingredients = set()
            
            for pref in user_preferences[:5]:  # Top 5 preferences
                menu_item = self._find_menu_item(pref.item_name)
                if menu_item:
                    user_dietary_prefs.update(menu_item.get('dietary_tags', []))
                    user_categories.add(menu_item.get('category', ''))
                    user_ingredients.update(menu_item.get('ingredients', []))
            
            # Score menu items based on content similarity
            item_scores = []
            for category in self.menu_data:
                category_name = category.get('category', '')
                
                for item in category.get('items', []):
                    # Skip items user has already ordered frequently
                    if any(pref.item_name.lower() == item.get('name', '').lower() 
                           for pref in user_preferences[:3]):
                        continue
                    
                    score = 0
                    
                    # Dietary similarity
                    item_dietary = set(item.get('dietary_tags', []))
                    dietary_similarity = len(user_dietary_prefs.intersection(item_dietary)) / max(1, len(user_dietary_prefs.union(item_dietary)))
                    score += dietary_similarity * 0.4
                    
                    # Category similarity
                    if category_name in user_categories:
                        score += 0.3
                    
                    # Ingredient similarity
                    item_ingredients = set(item.get('ingredients', []))
                    ingredient_similarity = len(user_ingredients.intersection(item_ingredients)) / max(1, len(user_ingredients.union(item_ingredients)))
                    score += ingredient_similarity * 0.2
                    
                    # Popularity bonus
                    score += item.get('popularity_score', 0.5) * 0.1
                    
                    if score > 0.2:
                        recommendation = RecommendationItem(
                            name=item.get('name', ''),
                            category=category_name,
                            price=item.get('price', 0),
                            confidence_score=score,
                            reason=f"Similar to items you've enjoyed before",
                            dietary_tags=item.get('dietary_tags', []),
                            nutrition_info=item.get('nutrition', {}),
                            estimated_preference=score
                        )
                        item_scores.append(recommendation)
            
            # Sort and return top k
            item_scores.sort(key=lambda x: x.confidence_score, reverse=True)
            recommendations = item_scores[:k]
            
        except Exception as e:
            print(f"Content-based filtering error: {e}")
            recommendations = self.get_popular_items(k)
        
        return recommendations
    
    def get_popular_items(self, k: int = 5) -> List[RecommendationItem]:
        """Get popular items as fallback recommendations"""
        popular_items = []
        
        for category in self.menu_data:
            category_name = category.get('category', '')
            
            for item in category.get('items', []):
                popularity = item.get('popularity_score', 0.5)
                
                if popularity > 0.7:  # High popularity threshold
                    recommendation = RecommendationItem(
                        name=item.get('name', ''),
                        category=category_name,
                        price=item.get('price', 0),
                        confidence_score=popularity,
                        reason="Popular choice among customers",
                        dietary_tags=item.get('dietary_tags', []),
                        nutrition_info=item.get('nutrition', {}),
                        estimated_preference=popularity
                    )
                    popular_items.append(recommendation)
        
        # Sort by popularity and return top k
        popular_items.sort(key=lambda x: x.confidence_score, reverse=True)
        return popular_items[:k]
    
    def hybrid_recommendations(self, session_id: str, k: int = 8) -> List[RecommendationItem]:
        """Generate hybrid recommendations combining multiple approaches"""
        try:
            # Get recommendations from different methods
            collaborative_recs = self.collaborative_filtering_recommendations(session_id, k//2)
            content_recs = self.content_based_recommendations(session_id, k//2)
            popular_recs = self.get_popular_items(k//4)
            
            # Combine and deduplicate
            all_recs = collaborative_recs + content_recs + popular_recs
            
            # Remove duplicates
            seen_names = set()
            unique_recs = []
            
            for rec in all_recs:
                if rec.name not in seen_names:
                    seen_names.add(rec.name)
                    unique_recs.append(rec)
            
            # Sort by confidence score
            unique_recs.sort(key=lambda x: x.confidence_score, reverse=True)
            
            return unique_recs[:k]
            
        except Exception as e:
            print(f"Hybrid recommendations error: {e}")
            return self.get_popular_items(k)
    
    def get_contextual_recommendations(self, 
                                    session_id: str, 
                                    context: str = "",
                                    dietary_prefs: List[str] = None,
                                    price_range: Tuple[float, float] = None,
                                    k: int = 5) -> List[RecommendationItem]:
        """Get contextual recommendations based on current situation"""
        
        # Start with hybrid recommendations
        recommendations = self.hybrid_recommendations(session_id, k * 2)
        
        # Apply filters
        filtered_recs = []
        
        for rec in recommendations:
            # Dietary filter
            if dietary_prefs:
                if not any(pref.lower() in [tag.lower() for tag in rec.dietary_tags] for pref in dietary_prefs):
                    continue
            
            # Price filter
            if price_range:
                min_price, max_price = price_range
                if not (min_price <= rec.price <= max_price):
                    continue
            
            # Context-based adjustments
            if context:
                context_lower = context.lower()
                
                # Time-based recommendations
                if any(word in context_lower for word in ['breakfast', 'morning']):
                    if rec.category.lower() in ['beverages', 'desserts']:
                        rec.confidence_score += 0.1
                elif any(word in context_lower for word in ['lunch', 'dinner']):
                    if rec.category.lower() in ['main courses']:
                        rec.confidence_score += 0.1
                elif any(word in context_lower for word in ['snack', 'light']):
                    if rec.category.lower() in ['appetizers', 'desserts']:
                        rec.confidence_score += 0.1
                
                # Mood-based recommendations
                if any(word in context_lower for word in ['healthy', 'diet', 'fitness']):
                    if 'vegan' in rec.dietary_tags or 'vegetarian' in rec.dietary_tags:
                        rec.confidence_score += 0.2
                elif any(word in context_lower for word in ['comfort', 'indulge', 'treat']):
                    if rec.category.lower() in ['desserts', 'main courses']:
                        rec.confidence_score += 0.1
            
            filtered_recs.append(rec)
        
        # Sort by adjusted confidence score
        filtered_recs.sort(key=lambda x: x.confidence_score, reverse=True)
        
        return filtered_recs[:k]
    
    def generate_recommendation_explanation(self, recommendations: List[RecommendationItem], session_id: str) -> str:
        """Generate natural language explanation for recommendations"""
        if not recommendations:
            return "I don't have enough information about your preferences yet. Try ordering a few items first!"
        
        if not self.llm:
            # Fallback explanation
            items = [f"{rec.name} (${rec.price:.2f})" for rec in recommendations[:3]]
            return f"Based on popular choices, I recommend: {', '.join(items)}"
        
        try:
            # Get user preferences for context
            user_prefs = self.extract_user_preferences(session_id)
            pref_summary = ""
            
            if user_prefs:
                top_categories = list(set(pref.category for pref in user_prefs[:3]))
                top_items = [pref.item_name for pref in user_prefs[:3]]
                pref_summary = f"You've previously enjoyed {', '.join(top_items)} from {', '.join(top_categories)} categories."
            
            # Prepare recommendation data
            rec_data = []
            for rec in recommendations[:5]:
                rec_data.append({
                    "name": rec.name,
                    "category": rec.category,
                    "price": rec.price,
                    "reason": rec.reason,
                    "dietary_tags": rec.dietary_tags
                })
            
            prompt = f"""
            Generate a friendly, personalized recommendation explanation for a food ordering customer.
            
            User's previous preferences: {pref_summary}
            
            Recommended items: {json.dumps(rec_data, indent=2)}
            
            Create a natural, conversational explanation that:
            1. Acknowledges their preferences if available
            2. Explains why these items are recommended
            3. Highlights key features (dietary, price, popularity)
            4. Encourages them to try something new
            5. Keeps it concise but engaging
            
            Response:
            """
            
            response = self.llm.invoke(prompt)
            return response.content if hasattr(response, 'content') else str(response)
            
        except Exception as e:
            print(f"Recommendation explanation generation failed: {e}")
            items = [f"{rec.name} (${rec.price:.2f})" for rec in recommendations[:3]]
            return f"I recommend trying: {', '.join(items)}. These are popular choices that match your preferences!"

# Global personalization engine
recommendation_engine = PersonalizationEngine()

def get_personalized_recommendations(session_id: str, 
                                   context: str = "",
                                   dietary_prefs: List[str] = None,
                                   price_range: Tuple[float, float] = None,
                                   k: int = 5) -> str:
    """Main entry point for getting personalized recommendations"""
    try:
        recommendations = recommendation_engine.get_contextual_recommendations(
            session_id=session_id,
            context=context,
            dietary_prefs=dietary_prefs,
            price_range=price_range,
            k=k
        )
        
        explanation = recommendation_engine.generate_recommendation_explanation(recommendations, session_id)
        
        # Log the recommendation
        log_conversation(session_id, "system", f"Generated {len(recommendations)} recommendations", {
            "context": context,
            "dietary_prefs": dietary_prefs,
            "price_range": price_range,
            "recommendation_count": len(recommendations)
        })
        
        return explanation
        
    except Exception as e:
        print(f"Personalized recommendations error: {e}")
        return "I'd be happy to recommend some popular items! Try our Margherita Pizza, Chicken Burger, or Fresh Lemonade - they're customer favorites!"

def analyze_user_preferences(session_id: str) -> Dict[str, Any]:
    """Analyze and return user preference summary"""
    try:
        preferences = recommendation_engine.extract_user_preferences(session_id)
        dietary_prefs = recommendation_engine.analyze_dietary_preferences(session_id)
        price_sensitivity = recommendation_engine.get_price_sensitivity(session_id)
        
        return {
            "top_items": [pref.item_name for pref in preferences[:5]],
            "preferred_categories": list(set(pref.category for pref in preferences[:5])),
            "dietary_preferences": dietary_prefs,
            "price_sensitivity": price_sensitivity,
            "total_orders": len(get_order_history(session_id)),
            "preference_strength": np.mean([pref.preference_score for pref in preferences]) if preferences else 0
        }
        
    except Exception as e:
        print(f"User preference analysis error: {e}")
        return {
            "top_items": [],
            "preferred_categories": [],
            "dietary_preferences": [],
            "price_sensitivity": 0.5,
            "total_orders": 0,
            "preference_strength": 0
        }