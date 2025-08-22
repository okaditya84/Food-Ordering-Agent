import os
import json
import re
import traceback
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum

import spacy
from pydantic import BaseModel, Field
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from dotenv import load_dotenv

from config import get_config, GROQ_API_KEY
from database import get_conversations, log_conversation

load_dotenv()

class IntentType(Enum):
    MENU_INQUIRY = "menu_inquiry"
    ORDER_ADD = "order_add"
    ORDER_REMOVE = "order_remove"
    ORDER_MODIFY = "order_modify"
    ORDER_CONFIRM = "order_confirm"
    ORDER_CANCEL = "order_cancel"
    CART_VIEW = "cart_view"
    CART_CLEAR = "cart_clear"
    HISTORY_VIEW = "history_view"
    RECOMMENDATIONS = "recommendations"
    DIETARY_INQUIRY = "dietary_inquiry"
    NUTRITIONAL_INFO = "nutritional_info"
    PRICE_INQUIRY = "price_inquiry"
    AVAILABILITY_CHECK = "availability_check"
    DELIVERY_INFO = "delivery_info"
    PAYMENT_INFO = "payment_info"
    CUSTOMER_SERVICE = "customer_service"
    COMPLAINT = "complaint"
    COMPLIMENT = "compliment"
    SMALLTALK = "smalltalk"
    HELP = "help"
    UNKNOWN = "unknown"

@dataclass
class Entity:
    text: str
    label: str
    start: int
    end: int
    confidence: float = 1.0
    normalized_value: Optional[str] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

@dataclass
class MenuItem:
    name: str
    quantity: int = 1
    modifiers: List[str] = None
    size: Optional[str] = None
    special_instructions: Optional[str] = None
    confidence: float = 1.0

    def __post_init__(self):
        if self.modifiers is None:
            self.modifiers = []

class NLUResult(BaseModel):
    intent: str
    confidence: float
    entities: List[Dict[str, Any]] = Field(default_factory=list)
    menu_items: List[Dict[str, Any]] = Field(default_factory=list)
    dietary_preferences: List[str] = Field(default_factory=list)
    price_range: Optional[Dict[str, float]] = None
    customer_info: Dict[str, str] = Field(default_factory=dict)
    sentiment: str = "neutral"
    urgency: str = "normal"
    context_references: List[str] = Field(default_factory=list)
    follow_up_questions: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)

class AdvancedNaturalLanguageProcessor:
    """Advanced natural language understanding system"""
    
    def __init__(self):
        self.config = get_config("nlu")
        self.llm = self._create_llm()
        self.nlp = self._load_spacy_model()
        self.parser = JsonOutputParser(pydantic_object=NLUResult)
        self.intent_patterns = self._load_intent_patterns()
        self.entity_patterns = self._load_entity_patterns()
        
    def _create_llm(self) -> Optional[ChatGroq]:
        """Create LLM instance for NLU processing"""
        try:
            return ChatGroq(
                model="llama3-70b-8192",
                groq_api_key=GROQ_API_KEY,
                temperature=0.1,
                max_tokens=2048
            )
        except Exception as e:
            print(f"Failed to create LLM: {e}")
            return None
    
    def _load_spacy_model(self):
        """Load spaCy model for entity extraction"""
        try:
            return spacy.load("en_core_web_sm")
        except OSError:
            print("spaCy model not found. Install with: python -m spacy download en_core_web_sm")
            return None
    
    def _load_intent_patterns(self) -> Dict[str, List[str]]:
        """Load intent classification patterns"""
        return {
            IntentType.MENU_INQUIRY.value: [
                r"\b(menu|show|display|list|what.*available|what.*have)\b",
                r"\b(food|items|dishes|options|choices)\b",
                r"\b(see.*menu|browse|look at)\b"
            ],
            IntentType.ORDER_ADD.value: [
                r"\b(add|order|get|want|need|buy|purchase)\b",
                r"\b(to.*cart|my.*order)\b",
                r"\b(\d+.*of|some|a|an)\b.*\b(pizza|burger|drink)\b"
            ],
            IntentType.ORDER_REMOVE.value: [
                r"\b(remove|delete|take.*out|cancel.*item)\b",
                r"\b(from.*cart|from.*order)\b",
                r"\b(don't.*want|changed.*mind)\b"
            ],
            IntentType.ORDER_CONFIRM.value: [
                r"\b(confirm|place|finalize|checkout|proceed)\b",
                r"\b(order|purchase|buy)\b",
                r"\b(ready.*order|complete.*order)\b"
            ],
            IntentType.RECOMMENDATIONS.value: [
                r"\b(recommend|suggest|popular|best|favorite)\b",
                r"\b(what.*good|what.*try|surprise.*me)\b",
                r"\b(chef.*special|today.*special)\b"
            ],
            IntentType.DIETARY_INQUIRY.value: [
                r"\b(vegetarian|vegan|gluten.*free|dairy.*free)\b",
                r"\b(allergic|allergy|dietary|restriction)\b",
                r"\b(keto|low.*carb|healthy|organic)\b"
            ],
            IntentType.NUTRITIONAL_INFO.value: [
                r"\b(calories|nutrition|protein|carbs|fat)\b",
                r"\b(healthy|diet|nutritional.*info)\b",
                r"\b(how.*many.*calories|nutritional.*value)\b"
            ]
        }
    
    def _load_entity_patterns(self) -> Dict[str, str]:
        """Load entity extraction patterns"""
        return {
            "QUANTITY": r"\b(\d+|one|two|three|four|five|six|seven|eight|nine|ten|a|an|some|few|several)\b",
            "FOOD_ITEM": r"\b(pizza|burger|pasta|salad|soup|sandwich|drink|coffee|tea|cake|ice.*cream)\b",
            "SIZE": r"\b(small|medium|large|extra.*large|xl|regular|mini)\b",
            "MODIFIER": r"\b(extra|no|without|with|add|remove|spicy|mild|hot|cold)\b",
            "PRICE": r"\$\d+\.?\d*|\b\d+\s*dollars?\b|\bunder\s*\$?\d+|\bless.*than\s*\$?\d+",
            "TIME": r"\b(now|asap|soon|later|tonight|today|tomorrow|in.*\d+.*minutes?)\b",
            "PHONE": r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b",
            "EMAIL": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
        }
    
    def extract_entities_spacy(self, text: str) -> List[Entity]:
        """Extract entities using spaCy NER"""
        entities = []
        if not self.nlp:
            return entities
            
        doc = self.nlp(text)
        for ent in doc.ents:
            entities.append(Entity(
                text=ent.text,
                label=ent.label_,
                start=ent.start_char,
                end=ent.end_char,
                confidence=0.8,
                metadata={"spacy_label": ent.label_}
            ))
        return entities
    
    def extract_entities_patterns(self, text: str) -> List[Entity]:
        """Extract entities using regex patterns"""
        entities = []
        for label, pattern in self.entity_patterns.items():
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                entities.append(Entity(
                    text=match.group(),
                    label=label,
                    start=match.start(),
                    end=match.end(),
                    confidence=0.7,
                    metadata={"pattern_based": True}
                ))
        return entities
    
    def extract_menu_items(self, text: str, menu_data: List[Dict]) -> List[MenuItem]:
        """Extract and normalize menu items from text"""
        items = []
        text_lower = text.lower()
        
        # Load menu items for matching
        menu_items = []
        for category in menu_data:
            for item in category.get("items", []):
                menu_items.append(item["name"].lower())
        
        # Extract quantities
        quantity_matches = re.findall(r'(\d+|one|two|three|four|five)\s+([a-zA-Z\s]+)', text_lower)
        
        for item_name in menu_items:
            if item_name in text_lower:
                # Find quantity
                quantity = 1
                for qty_match in quantity_matches:
                    qty_text, item_text = qty_match
                    if item_name in item_text.lower():
                        quantity = self._parse_quantity(qty_text)
                        break
                
                # Extract modifiers
                modifiers = []
                modifier_patterns = [
                    r'extra\s+\w+', r'no\s+\w+', r'without\s+\w+', 
                    r'with\s+\w+', r'spicy', r'mild', r'hot', r'cold'
                ]
                for pattern in modifier_patterns:
                    matches = re.findall(pattern, text_lower)
                    modifiers.extend(matches)
                
                items.append(MenuItem(
                    name=item_name,
                    quantity=quantity,
                    modifiers=modifiers,
                    confidence=0.8
                ))
        
        return items
    
    def _parse_quantity(self, qty_text: str) -> int:
        """Parse quantity from text"""
        qty_map = {
            "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
            "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10
        }
        
        if qty_text.isdigit():
            return int(qty_text)
        return qty_map.get(qty_text.lower(), 1)
    
    def classify_intent_patterns(self, text: str) -> Tuple[str, float]:
        """Classify intent using pattern matching"""
        text_lower = text.lower()
        best_intent = IntentType.UNKNOWN.value
        best_score = 0.0
        
        for intent, patterns in self.intent_patterns.items():
            score = 0.0
            for pattern in patterns:
                if re.search(pattern, text_lower):
                    score += 1.0
            
            # Normalize score
            if patterns:
                score = score / len(patterns)
            
            if score > best_score:
                best_score = score
                best_intent = intent
        
        return best_intent, best_score
    
    def get_conversation_context(self, session_id: str, window_size: int = 5) -> List[Dict]:
        """Get recent conversation context"""
        try:
            conversations = get_conversations(session_id, limit=window_size * 2)
            return conversations[:window_size]
        except Exception:
            return []
    
    def analyze_sentiment(self, text: str) -> str:
        """Analyze sentiment of the text"""
        positive_words = [
            "good", "great", "excellent", "amazing", "love", "like", "happy",
            "satisfied", "pleased", "wonderful", "fantastic", "awesome"
        ]
        negative_words = [
            "bad", "terrible", "awful", "hate", "dislike", "angry", "upset",
            "disappointed", "frustrated", "horrible", "disgusting", "worst"
        ]
        
        text_lower = text.lower()
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
        if positive_count > negative_count:
            return "positive"
        elif negative_count > positive_count:
            return "negative"
        else:
            return "neutral"
    
    def detect_urgency(self, text: str) -> str:
        """Detect urgency level from text"""
        urgent_words = [
            "urgent", "asap", "immediately", "now", "quickly", "fast",
            "emergency", "rush", "hurry", "soon"
        ]
        
        text_lower = text.lower()
        if any(word in text_lower for word in urgent_words):
            return "high"
        elif any(word in text_lower for word in ["later", "whenever", "no rush"]):
            return "low"
        else:
            return "normal"
    
    def extract_customer_info(self, text: str) -> Dict[str, str]:
        """Extract customer information from text"""
        info = {}
        
        # Phone number
        phone_match = re.search(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', text)
        if phone_match:
            info["phone"] = phone_match.group()
        
        # Email
        email_match = re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text)
        if email_match:
            info["email"] = email_match.group()
        
        # Name (simple heuristic)
        name_patterns = [
            r"my name is ([A-Za-z\s]+)",
            r"i'm ([A-Za-z\s]+)",
            r"this is ([A-Za-z\s]+)"
        ]
        for pattern in name_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                info["name"] = match.group(1).strip()
                break
        
        return info
    
    def generate_follow_up_questions(self, intent: str, entities: List[Entity]) -> List[str]:
        """Generate contextual follow-up questions"""
        questions = []
        
        if intent == IntentType.ORDER_ADD.value:
            if not any(e.label == "FOOD_ITEM" for e in entities):
                questions.append("What would you like to order?")
            if not any(e.label == "QUANTITY" for e in entities):
                questions.append("How many would you like?")
        
        elif intent == IntentType.DIETARY_INQUIRY.value:
            questions.extend([
                "Do you have any specific dietary restrictions?",
                "Are you looking for vegetarian, vegan, or gluten-free options?"
            ])
        
        elif intent == IntentType.RECOMMENDATIONS.value:
            questions.extend([
                "What type of cuisine do you prefer?",
                "Are you looking for something light or hearty?",
                "Do you have any dietary preferences?"
            ])
        
        return questions
    
    async def process_with_llm(self, text: str, context: List[Dict]) -> Dict[str, Any]:
        """Process text using LLM for advanced understanding"""
        if not self.llm:
            return {}
        
        context_str = ""
        if context:
            context_str = "\n".join([
                f"{msg['role']}: {msg['content']}" 
                for msg in context[-3:]  # Last 3 messages
            ])
        
        prompt = PromptTemplate(
            template="""
            Analyze the following user message in the context of a food ordering system.
            
            Previous conversation context:
            {context}
            
            Current user message: {text}
            
            Extract and return JSON with:
            - intent: one of [menu_inquiry, order_add, order_remove, order_confirm, recommendations, dietary_inquiry, nutritional_info, customer_service, smalltalk, help, unknown]
            - confidence: float 0-1
            - entities: list of {{text, label, confidence}}
            - menu_items: list of {{name, quantity, modifiers}}
            - dietary_preferences: list of dietary tags
            - customer_info: {{name, phone, email, address}}
            - sentiment: positive/negative/neutral
            - urgency: low/normal/high
            - context_references: references to previous messages
            - follow_up_questions: suggested questions to ask user
            
            {format_instructions}
            """,
            input_variables=["text", "context"],
            partial_variables={"format_instructions": self.parser.get_format_instructions()}
        )
        
        try:
            chain = prompt | self.llm | self.parser
            result = await chain.ainvoke({
                "text": text,
                "context": context_str
            })
            return result if isinstance(result, dict) else result.dict()
        except Exception as e:
            print(f"LLM processing failed: {e}")
            return {}
    
    def process(self, text: str, session_id: str, menu_data: List[Dict] = None) -> NLUResult:
        """Main processing function"""
        if not text or not text.strip():
            return NLUResult(
                intent=IntentType.HELP.value,
                confidence=0.0
            )
        
        # Get conversation context
        context = self.get_conversation_context(session_id)
        
        # Pattern-based intent classification
        intent, confidence = self.classify_intent_patterns(text)
        
        # Entity extraction
        spacy_entities = self.extract_entities_spacy(text)
        pattern_entities = self.extract_entities_patterns(text)
        all_entities = spacy_entities + pattern_entities
        
        # Menu item extraction
        menu_items = []
        if menu_data:
            menu_items = self.extract_menu_items(text, menu_data)
        
        # Additional analysis
        sentiment = self.analyze_sentiment(text)
        urgency = self.detect_urgency(text)
        customer_info = self.extract_customer_info(text)
        
        # Generate follow-up questions
        follow_up_questions = self.generate_follow_up_questions(intent, all_entities)
        
        # Create result
        result = NLUResult(
            intent=intent,
            confidence=confidence,
            entities=[asdict(e) for e in all_entities],
            menu_items=[asdict(item) for item in menu_items],
            customer_info=customer_info,
            sentiment=sentiment,
            urgency=urgency,
            follow_up_questions=follow_up_questions,
            metadata={
                "processing_time": datetime.now().isoformat(),
                "context_length": len(context),
                "entity_count": len(all_entities)
            }
        )
        
        return result

# Global NLU instance
nlp_processor = AdvancedNaturalLanguageProcessor()

def process_user_input(text: str, session_id: str, menu_data: List[Dict] = None) -> Dict[str, Any]:
    """Main entry point for NLU processing"""
    try:
        result = nlp_processor.process(text, session_id, menu_data)
        
        # Log the conversation
        log_conversation(session_id, "user", text, {
            "intent": result.intent,
            "confidence": result.confidence,
            "entities": len(result.entities)
        })
        
        return result.dict()
    
    except Exception as e:
        print(f"NLU processing error: {e}")
        traceback.print_exc()
        
        # Fallback result
        return {
            "intent": IntentType.HELP.value,
            "confidence": 0.0,
            "entities": [],
            "menu_items": [],
            "sentiment": "neutral",
            "urgency": "normal",
            "error": str(e)
        }