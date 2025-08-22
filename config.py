import os
from typing import Dict, Any
from dotenv import load_dotenv

load_dotenv()

# API Configuration
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Model Configuration
DEFAULT_LLM_MODEL = "llama-3.3-70b-versatile"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L12-v2"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-12-v2"

# Agent Configuration
AGENT_CONFIG = {
    "temperature": 0.1,
    "max_tokens": 2048,
    "top_p": 0.9,
    "frequency_penalty": 0.1,
    "presence_penalty": 0.1
}

# NLU Configuration
NLU_CONFIG = {
    "confidence_threshold": 0.7,
    "max_entities": 10,
    "context_window": 5,  # Number of previous messages to consider
    "intent_categories": [
        "menu_inquiry", "order_management", "cart_operations", 
        "customer_service", "recommendations", "dietary_preferences",
        "nutritional_info", "order_tracking", "payment", "delivery"
    ]
}

# RAG Configuration
RAG_CONFIG = {
    "chunk_size": 512,
    "chunk_overlap": 50,
    "top_k_retrieval": 15,
    "rerank_top_k": 8,
    "similarity_threshold": 0.6,
    "query_expansion": True,
    "use_reranking": True
}

# Database Configuration
DATABASE_CONFIG = {
    "db_path": "orders.db",
    "backup_interval": 3600,  # seconds
    "max_conversation_history": 1000,
    "session_timeout": 7200  # 2 hours
}

# UI Configuration
UI_CONFIG = {
    "page_title": "🍽️ Food Ordering Agent",
    "theme": "light",
    "sidebar_width": 400,
    "chat_height": 600,
    "enable_voice": True,
    "enable_images": True,
    "auto_scroll": True
}

# Business Logic Configuration
BUSINESS_CONFIG = {
    "cart_timeout": 3600, 
    "max_cart_items": 50,
    "delivery_fee": 2.99,
    "tax_rate": 0.08,
    "promo_codes": {
        "SAVE10": {"type": "percentage", "value": 0.10, "min_order": 15.0},
        "FIRST20": {"type": "percentage", "value": 0.20, "min_order": 25.0},
        "FREE5": {"type": "fixed", "value": 5.0, "min_order": 30.0}
    },
    "dietary_filters": [
        "vegetarian", "vegan", "gluten_free", "dairy_free", 
        "nut_free", "low_carb", "keto", "halal", "kosher"
    ]
}

# Personalization Configuration
PERSONALIZATION_CONFIG = {
    "enable_recommendations": True,
    "learning_rate": 0.1,
    "min_orders_for_personalization": 3,
    "recommendation_algorithms": ["collaborative", "content_based", "hybrid"],
    "preference_decay": 0.95,
    "max_user_preferences": 20
}

# Advanced Features Configuration
ADVANCED_FEATURES = {
    "nutritional_analysis": True,
    "allergen_detection": True,
    "smart_substitutions": True,
    "order_optimization": True,
    "predictive_ordering": True,
    "sentiment_analysis": True,
    "multi_language": False,  
    "voice_ordering": True,
    "image_recognition": False 
}

# Logging Configuration
LOGGING_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "file": "food_agent.log",
    "max_size": "10MB",
    "backup_count": 5
}

# Error Handling Configuration
ERROR_CONFIG = {
    "max_retries": 3,
    "retry_delay": 1.0,
    "fallback_responses": {
        "llm_error": "I'm having trouble processing that right now. Could you please rephrase your request?",
        "menu_error": "I'm having trouble accessing the menu. Please try again in a moment.",
        "order_error": "There was an issue with your order. Please contact support if this continues.",
        "general_error": "Something went wrong. Let me try to help you in a different way."
    }
}

# Performance Configuration
PERFORMANCE_CONFIG = {
    "cache_ttl": 300,
    "max_concurrent_requests": 10,
    "request_timeout": 30,
    "enable_caching": True,
    "enable_compression": True
}

def get_config(section: str = None) -> Dict[str, Any]:
    """Get configuration for a specific section or all configurations"""
    configs = {
        "agent": AGENT_CONFIG,
        "nlu": NLU_CONFIG,
        "rag": RAG_CONFIG,
        "database": DATABASE_CONFIG,
        "ui": UI_CONFIG,
        "business": BUSINESS_CONFIG,
        "personalization": PERSONALIZATION_CONFIG,
        "features": ADVANCED_FEATURES,
        "logging": LOGGING_CONFIG,
        "error": ERROR_CONFIG,
        "performance": PERFORMANCE_CONFIG
    }
    
    if section:
        return configs.get(section, {})
    return configs

def validate_config() -> bool:
    """Validate that required configurations are present"""
    required_keys = ["GROQ_API_KEY"]
    missing_keys = [key for key in required_keys if not globals().get(key)]
    
    if missing_keys:
        print(f"Missing required configuration keys: {missing_keys}")
        return False
    return True