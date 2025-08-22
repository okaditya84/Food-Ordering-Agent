import logging
import traceback
import functools
from typing import Any, Callable, Dict, List, Union
from datetime import datetime
from enum import Enum
import json

from config import ERROR_CONFIG, LOGGING_CONFIG

# Configure logging
logging.basicConfig(
    level=getattr(logging, LOGGING_CONFIG.get("level", "INFO")),
    format=LOGGING_CONFIG.get("format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s"),
    handlers=[
        logging.FileHandler(LOGGING_CONFIG.get("file", "food_agent.log")),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

class ErrorType(Enum):
    """Error type classification"""
    LLM_ERROR = "llm_error"
    DATABASE_ERROR = "database_error"
    MENU_ERROR = "menu_error"
    ORDER_ERROR = "order_error"
    NETWORK_ERROR = "network_error"
    VALIDATION_ERROR = "validation_error"
    PARSING_ERROR = "parsing_error"
    TIMEOUT_ERROR = "timeout_error"
    AUTHENTICATION_ERROR = "auth_error"
    RATE_LIMIT_ERROR = "rate_limit_error"
    GENERAL_ERROR = "general_error"

class ErrorSeverity(Enum):
    """Error severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class FoodAgentError(Exception):
    """Base exception for food agent errors"""
    
    def __init__(self, 
                 message: str, 
                 error_type: ErrorType = ErrorType.GENERAL_ERROR,
                 severity: ErrorSeverity = ErrorSeverity.MEDIUM,
                 context: Dict[str, Any] = None,
                 original_error: Exception = None):
        super().__init__(message)
        self.message = message
        self.error_type = error_type
        self.severity = severity
        self.context = context or {}
        self.original_error = original_error
        self.timestamp = datetime.now()

class ErrorHandler:
    """Centralized error handling system"""
    
    def __init__(self):
        self.config = ERROR_CONFIG
        self.error_counts = {}
        self.fallback_responses = self.config.get("fallback_responses", {})
        
    def log_error(self, error: Union[Exception, FoodAgentError], context: Dict[str, Any] = None):
        """Log error with context information"""
        try:
            if isinstance(error, FoodAgentError):
                error_info = {
                    "type": error.error_type.value,
                    "severity": error.severity.value,
                    "message": error.message,
                    "context": error.context,
                    "timestamp": error.timestamp.isoformat()
                }
                
                if error.original_error:
                    error_info["original_error"] = str(error.original_error)
                    error_info["traceback"] = traceback.format_exception(
                        type(error.original_error), 
                        error.original_error, 
                        error.original_error.__traceback__
                    )
            else:
                error_info = {
                    "type": ErrorType.GENERAL_ERROR.value,
                    "severity": ErrorSeverity.MEDIUM.value,
                    "message": str(error),
                    "context": context or {},
                    "timestamp": datetime.now().isoformat(),
                    "traceback": traceback.format_exception(type(error), error, error.__traceback__)
                }
            
            # Log based on severity
            if isinstance(error, FoodAgentError):
                if error.severity == ErrorSeverity.CRITICAL:
                    logger.critical(f"CRITICAL ERROR: {json.dumps(error_info, indent=2)}")
                elif error.severity == ErrorSeverity.HIGH:
                    logger.error(f"HIGH SEVERITY ERROR: {json.dumps(error_info, indent=2)}")
                elif error.severity == ErrorSeverity.MEDIUM:
                    logger.warning(f"MEDIUM SEVERITY ERROR: {json.dumps(error_info, indent=2)}")
                else:
                    logger.info(f"LOW SEVERITY ERROR: {json.dumps(error_info, indent=2)}")
            else:
                logger.error(f"UNHANDLED ERROR: {json.dumps(error_info, indent=2)}")
            
            # Track error counts for monitoring
            error_type = error_info["type"]
            self.error_counts[error_type] = self.error_counts.get(error_type, 0) + 1
            
        except Exception as logging_error:
            # Fallback logging if main logging fails
            print(f"LOGGING ERROR: {logging_error}")
            print(f"ORIGINAL ERROR: {error}")
    
    def get_fallback_response(self, error_type: ErrorType, context: Dict[str, Any] = None) -> str:
        """Get appropriate fallback response for error type"""
        fallback_key = error_type.value
        
        # Get base fallback response
        base_response = self.fallback_responses.get(fallback_key, 
                                                   self.fallback_responses.get("general_error", 
                                                                             "I'm experiencing some technical difficulties. Please try again."))
        
        # Customize response based on context
        if context:
            session_id = context.get("session_id")
            user_input = context.get("user_input", "")
            
            if error_type == ErrorType.MENU_ERROR:
                if "search" in user_input.lower() or "menu" in user_input.lower():
                    return "I'm having trouble accessing the menu right now. You can try asking about specific items or categories, or try again in a moment."
            
            elif error_type == ErrorType.ORDER_ERROR:
                if "add" in user_input.lower() or "order" in user_input.lower():
                    return "I'm having trouble processing your order right now. Please try adding items one at a time, or contact support if this continues."
            
            elif error_type == ErrorType.LLM_ERROR:
                return "I'm having trouble understanding your request right now. Could you please rephrase it or try a simpler question?"
        
        return base_response
    
    def should_retry(self, error: Union[Exception, FoodAgentError], attempt: int) -> bool:
        """Determine if operation should be retried"""
        max_retries = self.config.get("max_retries", 3)
        
        if attempt >= max_retries:
            return False
        
        # Don't retry certain error types
        no_retry_types = [
            ErrorType.VALIDATION_ERROR,
            ErrorType.AUTHENTICATION_ERROR,
            ErrorType.PARSING_ERROR
        ]
        
        if isinstance(error, FoodAgentError) and error.error_type in no_retry_types:
            return False
        
        return True
    
    def get_retry_delay(self, attempt: int) -> float:
        """Get delay before retry (exponential backoff)"""
        base_delay = self.config.get("retry_delay", 1.0)
        return base_delay * (2 ** attempt)

# Global error handler instance
error_handler = ErrorHandler()

def handle_errors(error_type: ErrorType = ErrorType.GENERAL_ERROR, 
                 severity: ErrorSeverity = ErrorSeverity.MEDIUM,
                 fallback_response: str = None):
    """Decorator for handling errors in functions"""
    
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            max_retries = error_handler.config.get("max_retries", 3)
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                
                except FoodAgentError as e:
                    error_handler.log_error(e)
                    
                    if not error_handler.should_retry(e, attempt):
                        response = fallback_response or error_handler.get_fallback_response(
                            e.error_type, 
                            {"function": func.__name__, "args": args, "kwargs": kwargs}
                        )
                        return response
                    
                    if attempt < max_retries:
                        import time
                        time.sleep(error_handler.get_retry_delay(attempt))
                
                except Exception as e:
                    # Convert to FoodAgentError
                    food_error = FoodAgentError(
                        message=str(e),
                        error_type=error_type,
                        severity=severity,
                        context={"function": func.__name__, "args": str(args), "kwargs": str(kwargs)},
                        original_error=e
                    )
                    
                    error_handler.log_error(food_error)
                    
                    if not error_handler.should_retry(food_error, attempt):
                        response = fallback_response or error_handler.get_fallback_response(
                            error_type,
                            {"function": func.__name__, "args": args, "kwargs": kwargs}
                        )
                        return response
                    
                    if attempt < max_retries:
                        import time
                        time.sleep(error_handler.get_retry_delay(attempt))
            
            # If all retries failed
            response = fallback_response or error_handler.get_fallback_response(error_type)
            return response
        
        return wrapper
    return decorator

class FallbackManager:
    """Manages fallback mechanisms for different components"""
    
    def __init__(self):
        self.fallback_strategies = {
            "llm": self._llm_fallback,
            "database": self._database_fallback,
            "menu_search": self._menu_search_fallback,
            "recommendations": self._recommendations_fallback,
            "nlu": self._nlu_fallback
        }
    
    def _llm_fallback(self, context: Dict[str, Any]) -> str:
        """Fallback for LLM failures"""
        user_input = context.get("user_input", "").lower()
        
        # Simple keyword-based responses
        if any(word in user_input for word in ["menu", "show", "list"]):
            return "Here are some popular items: Margherita Pizza ($12.99), Chicken Burger ($9.99), Pasta Alfredo ($11.49). What would you like to know more about?"
        
        elif any(word in user_input for word in ["add", "order", "want"]):
            return "I'd be happy to help you order! Please tell me the specific item name and quantity, like 'add 2 margherita pizzas'."
        
        elif any(word in user_input for word in ["recommend", "suggest"]):
            return "I recommend trying our popular items: Margherita Pizza, Chicken Wings, or Fresh Lemonade. They're customer favorites!"
        
        elif any(word in user_input for word in ["cart", "order"]):
            return "You can view your cart by saying 'show my cart' or add items by saying 'add [item name]'."
        
        else:
            return "I'm here to help you order food! You can ask me to show the menu, add items to your cart, or get recommendations."
    
    def _database_fallback(self, context: Dict[str, Any]) -> Any:
        """Fallback for database failures"""
        operation = context.get("operation", "")
        
        if operation == "get_cart":
            return []  # Empty cart
        elif operation == "get_orders":
            return []  # No order history
        elif operation == "save_cart":
            return True  # Pretend success
        else:
            return None
    
    def _menu_search_fallback(self, context: Dict[str, Any]) -> str:
        """Fallback for menu search failures"""
        query = context.get("query", "").lower()
        
        # Basic menu information
        menu_info = {
            "pizza": "Margherita Pizza - $12.99: Classic pizza with fresh tomato sauce, mozzarella, and basil.",
            "burger": "Chicken Burger - $9.99: Grilled chicken patty with lettuce, tomato, onion, and mayo.",
            "pasta": "Pasta Alfredo - $11.49: Fettuccine pasta in a creamy parmesan sauce with garlic.",
            "drink": "Fresh Lemonade - $3.49: Homemade lemonade with mint.",
            "dessert": "Chocolate Cake - $6.99: Rich moist chocolate layer cake with fudge frosting."
        }
        
        for key, info in menu_info.items():
            if key in query:
                return f"Here's what I found: {info}"
        
        return "I'm having trouble searching the menu right now. Here are some popular items: Margherita Pizza ($12.99), Chicken Burger ($9.99), Pasta Alfredo ($11.49)."
    
    def _recommendations_fallback(self, context: Dict[str, Any]) -> str:
        """Fallback for recommendation failures"""
        return "I recommend trying our customer favorites: Margherita Pizza, Chicken Wings, and Fresh Lemonade. They're delicious and popular choices!"
    
    def _nlu_fallback(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback for NLU failures"""
        user_input = context.get("user_input", "").lower()
        
        # Simple intent classification
        if any(word in user_input for word in ["menu", "show", "list", "what"]):
            intent = "menu_inquiry"
        elif any(word in user_input for word in ["add", "order", "want", "get"]):
            intent = "order_add"
        elif any(word in user_input for word in ["remove", "delete", "cancel"]):
            intent = "order_remove"
        elif any(word in user_input for word in ["recommend", "suggest", "popular"]):
            intent = "recommendations"
        elif any(word in user_input for word in ["cart", "basket"]):
            intent = "cart_view"
        elif any(word in user_input for word in ["history", "orders", "past"]):
            intent = "history_view"
        else:
            intent = "help"
        
        return {
            "intent": intent,
            "confidence": 0.5,
            "entities": [],
            "menu_items": [],
            "sentiment": "neutral",
            "urgency": "normal"
        }
    
    def get_fallback(self, component: str, context: Dict[str, Any]) -> Any:
        """Get fallback for a specific component"""
        fallback_func = self.fallback_strategies.get(component)
        if fallback_func:
            try:
                return fallback_func(context)
            except Exception as e:
                logger.error(f"Fallback failed for {component}: {e}")
                return None
        return None

# Global fallback manager
fallback_manager = FallbackManager()

class HealthChecker:
    """System health monitoring and recovery"""
    
    def __init__(self):
        self.health_status = {
            "llm": True,
            "database": True,
            "menu_search": True,
            "recommendations": True,
            "overall": True
        }
        self.last_check = datetime.now()
    
    def check_llm_health(self) -> bool:
        """Check LLM service health"""
        try:
            from langchain_groq import ChatGroq
            from config import GROQ_API_KEY
            
            llm = ChatGroq(
                model="llama3-70b-8192",
                groq_api_key=GROQ_API_KEY,
                temperature=0.1,
                max_tokens=50
            )
            
            response = llm.invoke("Hello")
            return bool(response)
        
        except Exception as e:
            logger.warning(f"LLM health check failed: {e}")
            return False
    
    def check_database_health(self) -> bool:
        """Check database connectivity"""
        try:
            from database import get_cart
            get_cart("health_check")
            return True
        except Exception as e:
            logger.warning(f"Database health check failed: {e}")
            return False
    
    def check_menu_search_health(self) -> bool:
        """Check menu search functionality"""
        try:
            from intelligent_rag import search_menu
            result = search_menu("pizza", k=1)
            return bool(result)
        except Exception as e:
            logger.warning(f"Menu search health check failed: {e}")
            return False
    
    def run_health_check(self) -> Dict[str, bool]:
        """Run comprehensive health check"""
        try:
            self.health_status["llm"] = self.check_llm_health()
            self.health_status["database"] = self.check_database_health()
            self.health_status["menu_search"] = self.check_menu_search_health()
            
            # Overall health
            self.health_status["overall"] = all([
                self.health_status["llm"],
                self.health_status["database"],
                self.health_status["menu_search"]
            ])
            
            self.last_check = datetime.now()
            
            if not self.health_status["overall"]:
                logger.warning(f"System health issues detected: {self.health_status}")
            
            return self.health_status
        
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {"overall": False, "error": str(e)}
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get current health status"""
        return {
            "status": self.health_status,
            "last_check": self.last_check.isoformat(),
            "uptime_minutes": (datetime.now() - self.last_check).total_seconds() / 60
        }

# Global health checker
health_checker = HealthChecker()

def safe_execute(func: Callable, 
                fallback_component: str = None,
                context: Dict[str, Any] = None,
                default_return: Any = None) -> Any:
    """Safely execute a function with fallback support"""
    try:
        return func()
    except Exception as e:
        error_handler.log_error(e, context)
        
        if fallback_component:
            fallback_result = fallback_manager.get_fallback(fallback_component, context or {})
            if fallback_result is not None:
                return fallback_result
        
        return default_return

def validate_input(data: Any, validation_rules: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """Validate input data against rules"""
    errors = []
    
    try:
        # Required fields
        required_fields = validation_rules.get("required", [])
        if isinstance(data, dict):
            for field in required_fields:
                if field not in data or not data[field]:
                    errors.append(f"Required field '{field}' is missing or empty")
        
        # Type validation
        type_rules = validation_rules.get("types", {})
        if isinstance(data, dict):
            for field, expected_type in type_rules.items():
                if field in data and not isinstance(data[field], expected_type):
                    errors.append(f"Field '{field}' must be of type {expected_type.__name__}")
        
        # Custom validation
        custom_validator = validation_rules.get("custom")
        if custom_validator and callable(custom_validator):
            custom_errors = custom_validator(data)
            if custom_errors:
                errors.extend(custom_errors)
        
        return len(errors) == 0, errors
    
    except Exception as e:
        errors.append(f"Validation error: {str(e)}")
        return False, errors

# Export main components
__all__ = [
    'FoodAgentError',
    'ErrorType',
    'ErrorSeverity',
    'handle_errors',
    'error_handler',
    'fallback_manager',
    'health_checker',
    'safe_execute',
    'validate_input'
]