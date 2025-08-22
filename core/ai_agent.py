import json
import asyncio
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass, asdict
from enum import Enum

from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferWindowMemory

from config import get_config, GROQ_API_KEY
from database import log_conversation, get_conversations, log_api_usage
from core.natural_language_processor import process_user_input, IntentType
from core.knowledge_retrieval import search_menu
from core.recommendation_engine import get_personalized_recommendations, analyze_user_preferences
from core.intelligent_tools import SMART_TOOLS

class AgentState(Enum):
    LISTENING = "listening"
    PROCESSING = "processing"
    TOOL_CALLING = "tool_calling"
    RESPONDING = "responding"
    ERROR = "error"

@dataclass
class ConversationContext:
    session_id: str
    user_preferences: Dict[str, Any]
    current_cart: List[Dict]
    conversation_history: List[Dict]
    dietary_restrictions: List[str]
    price_sensitivity: float
    last_intent: str
    context_summary: str

class IntelligentFoodAgent:
    """Main intelligent agent with advanced reasoning capabilities"""
    
    def __init__(self):
        self.config = get_config("agent")
        self.llm = self._create_llm()
        self.memory = ConversationBufferWindowMemory(
            k=10,  # Remember last 10 exchanges
            return_messages=True,
            memory_key="chat_history"
        )
        self.agent_executor = self._create_agent()
        self.state = AgentState.LISTENING
        self.context_cache = {}  # Cache conversation contexts
        
    def _create_llm(self) -> ChatGroq:
        """Create the main LLM for the agent"""
        return ChatGroq(
            model="llama3-70b-8192",
            groq_api_key=GROQ_API_KEY,
            temperature=self.config.get("temperature", 0.1),
            max_tokens=self.config.get("max_tokens", 2048)
        )
    
    def _create_agent(self) -> AgentExecutor:
        """Create the agent executor with tools and prompts"""
        
        # Create the system prompt
        system_prompt = """
        You are an intelligent food ordering assistant with advanced reasoning capabilities. 
        Your goal is to provide exceptional customer service while helping users discover, order, and enjoy food.

        CORE CAPABILITIES:
        1. Natural language understanding of food-related queries
        2. Intelligent menu search and recommendations
        3. Smart cart management with optimization suggestions
        4. Personalized recommendations based on user history
        5. Dietary restriction awareness and validation
        6. Nutritional information and health-conscious suggestions
        7. Order optimization for value and satisfaction
        8. Context-aware conversation management

        PERSONALITY TRAITS:
        - Friendly, helpful, and enthusiastic about food
        - Knowledgeable about menu items, ingredients, and nutrition
        - Proactive in making suggestions and optimizations
        - Patient and understanding with dietary restrictions
        - Professional but conversational tone

        DECISION MAKING PROCESS:
        1. Understand user intent using context and conversation history
        2. Determine if tools are needed to fulfill the request
        3. Use appropriate tools with intelligent parameter selection
        4. Synthesize information from multiple sources when needed
        5. Provide comprehensive, helpful responses
        6. Proactively suggest improvements or alternatives

        TOOL USAGE GUIDELINES:
        - Use intelligent_menu_search for menu-related queries
        - Use smart_add_to_cart for adding items with validation
        - Use get_smart_recommendations for personalized suggestions
        - Use analyze_nutritional_info for health-related queries
        - Use optimize_current_order to improve user's cart
        - Always consider user's dietary preferences and restrictions

        RESPONSE GUIDELINES:
        - Be conversational and engaging
        - Provide specific, actionable information
        - Include prices, dietary info, and descriptions when relevant
        - Suggest alternatives when exact requests aren't available
        - Proactively offer complementary items or optimizations
        - Ask clarifying questions when needed
        - Acknowledge user preferences and history

        Remember: You're not just taking orders - you're helping users discover great food experiences!
        """
        
        # Create the prompt template
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])
        
        # Create the agent
        agent = create_tool_calling_agent(
            llm=self.llm,
            tools=SMART_TOOLS,
            prompt=prompt
        )
        
        # Create the executor
        return AgentExecutor(
            agent=agent,
            tools=SMART_TOOLS,
            memory=self.memory,
            verbose=True,
            max_iterations=3,
            early_stopping_method="generate",
            handle_parsing_errors=True
        )
    
    def _get_conversation_context(self, session_id: str) -> ConversationContext:
        """Get or create conversation context for a session"""
        if session_id in self.context_cache:
            # Update cache with fresh data
            context = self.context_cache[session_id]
            context.user_preferences = analyze_user_preferences(session_id)
            return context
        
        # Create new context
        from database import get_cart
        
        context = ConversationContext(
            session_id=session_id,
            user_preferences=analyze_user_preferences(session_id),
            current_cart=get_cart(session_id),
            conversation_history=get_conversations(session_id, limit=10),
            dietary_restrictions=analyze_user_preferences(session_id).get('dietary_preferences', []),
            price_sensitivity=analyze_user_preferences(session_id).get('price_sensitivity', 0.5),
            last_intent="",
            context_summary=""
        )
        
        self.context_cache[session_id] = context
        return context
    
    def _update_context_summary(self, context: ConversationContext, user_input: str, response: str):
        """Update the context summary with key information"""
        try:
            # Use LLM to create/update context summary
            if not context.context_summary:
                context.context_summary = f"User is interested in food ordering. Recent interaction: {user_input[:100]}..."
            else:
                # Update existing summary
                prompt = f"""
                Current context summary: {context.context_summary}
                
                Latest user input: {user_input}
                Agent response: {response[:200]}...
                
                Update the context summary to include key information about:
                - User preferences and dietary restrictions
                - Items they're interested in or have ordered
                - Any specific requests or constraints
                - Conversation flow and current focus
                
                Keep it concise (max 200 words) and focus on actionable information.
                """
                
                try:
                    summary_response = self.llm.invoke(prompt)
                    context.context_summary = summary_response.content if hasattr(summary_response, 'content') else str(summary_response)
                except Exception:
                    # Fallback: simple append
                    context.context_summary += f" Recent: {user_input[:50]}..."
                    if len(context.context_summary) > 500:
                        context.context_summary = context.context_summary[-400:]  # Keep last 400 chars
        
        except Exception as e:
            print(f"Context summary update failed: {e}")
    
    def _preprocess_input(self, user_input: str, session_id: str) -> Dict[str, Any]:
        """Preprocess user input with NLU and context analysis"""
        try:
            # Get NLU analysis
            nlu_result = process_user_input(user_input, session_id)
            
            # Get conversation context
            context = self._get_conversation_context(session_id)
            
            # Enhance input with context
            enhanced_input = {
                "original_input": user_input,
                "intent": nlu_result.get("intent", "unknown"),
                "confidence": nlu_result.get("confidence", 0.0),
                "entities": nlu_result.get("entities", []),
                "menu_items": nlu_result.get("menu_items", []),
                "sentiment": nlu_result.get("sentiment", "neutral"),
                "urgency": nlu_result.get("urgency", "normal"),
                "context": context,
                "session_id": session_id
            }
            
            return enhanced_input
            
        except Exception as e:
            print(f"Input preprocessing failed: {e}")
            return {
                "original_input": user_input,
                "intent": "unknown",
                "confidence": 0.0,
                "entities": [],
                "menu_items": [],
                "sentiment": "neutral",
                "urgency": "normal",
                "context": self._get_conversation_context(session_id),
                "session_id": session_id
            }
    
    def _should_use_tools(self, enhanced_input: Dict[str, Any]) -> bool:
        """Determine if tools should be used based on intent and context"""
        intent = enhanced_input.get("intent", "")
        confidence = enhanced_input.get("confidence", 0.0)
        
        # High-confidence intents that typically need tools
        tool_requiring_intents = [
            IntentType.MENU_INQUIRY.value,
            IntentType.ORDER_ADD.value,
            IntentType.ORDER_REMOVE.value,
            IntentType.ORDER_CONFIRM.value,
            IntentType.RECOMMENDATIONS.value,
            IntentType.NUTRITIONAL_INFO.value,
            IntentType.CART_VIEW.value,
            IntentType.HISTORY_VIEW.value
        ]
        
        # Use tools for high-confidence specific intents
        if intent in tool_requiring_intents and confidence > 0.6:
            return True
        
        # Use tools if menu items are mentioned
        if enhanced_input.get("menu_items"):
            return True
        
        # Use tools for complex queries (multiple entities)
        if len(enhanced_input.get("entities", [])) > 2:
            return True
        
        return False
    
    def _generate_direct_response(self, enhanced_input: Dict[str, Any]) -> str:
        """Generate direct response without tools for simple queries"""
        user_input = enhanced_input["original_input"]
        intent = enhanced_input.get("intent", "")
        context = enhanced_input.get("context")
        
        # Create context-aware prompt
        context_info = ""
        if context and context.user_preferences:
            prefs = context.user_preferences
            if prefs.get("preferred_categories"):
                context_info += f"User typically orders: {', '.join(prefs['preferred_categories'])}. "
            if prefs.get("dietary_preferences"):
                context_info += f"Dietary preferences: {', '.join(prefs['dietary_preferences'])}. "
        
        prompt = f"""
        User message: "{user_input}"
        Intent: {intent}
        Context: {context_info}
        
        Respond as a friendly food ordering assistant. Be helpful, conversational, and specific.
        If you need more information to help them, ask clarifying questions.
        If they're asking about menu items, suggest they be more specific so you can search properly.
        """
        
        try:
            response = self.llm.invoke(prompt)
            return response.content if hasattr(response, 'content') else str(response)
        except Exception as e:
            return "I'm here to help you with your food order! What would you like to know about our menu or how can I assist you today?"
    
    async def process_message(self, user_input: str, session_id: str) -> str:
        """Main message processing method"""
        self.state = AgentState.PROCESSING
        
        try:
            # Log user message
            log_conversation(session_id, "user", user_input)
            
            # Preprocess input
            enhanced_input = self._preprocess_input(user_input, session_id)
            
            # Determine processing approach
            if self._should_use_tools(enhanced_input):
                self.state = AgentState.TOOL_CALLING
                
                # Prepare input for agent with context
                context = enhanced_input.get("context")
                context_str = ""
                
                if context:
                    if context.dietary_restrictions:
                        context_str += f"User dietary preferences: {', '.join(context.dietary_restrictions)}. "
                    if context.current_cart:
                        cart_items = [f"{item['quantity']}x {item['name']}" for item in context.current_cart]
                        context_str += f"Current cart: {', '.join(cart_items)}. "
                    if context.context_summary:
                        context_str += f"Context: {context.context_summary}"
                
                # Enhanced input for agent
                agent_input = f"{user_input}\n\nSession ID: {session_id}"
                if context_str:
                    agent_input += f"\nContext: {context_str}"
                
                # Run agent with tools
                result = await self.agent_executor.ainvoke({
                    "input": agent_input,
                    "session_id": session_id
                })
                
                response = result.get("output", "I'm having trouble processing your request right now.")
                
            else:
                self.state = AgentState.RESPONDING
                # Generate direct response
                response = self._generate_direct_response(enhanced_input)
            
            # Update context
            context = enhanced_input.get("context")
            if context:
                context.last_intent = enhanced_input.get("intent", "")
                self._update_context_summary(context, user_input, response)
            
            # Log assistant response
            log_conversation(session_id, "assistant", response, {
                "intent": enhanced_input.get("intent"),
                "confidence": enhanced_input.get("confidence"),
                "used_tools": self.state == AgentState.TOOL_CALLING
            })
            
            self.state = AgentState.LISTENING
            return response
            
        except Exception as e:
            self.state = AgentState.ERROR
            error_msg = f"I apologize, but I encountered an error while processing your request. Please try again or rephrase your question."
            
            # Log error
            log_conversation(session_id, "system", f"Error: {str(e)}")
            
            print(f"Agent processing error: {e}")
            return error_msg
    
    def get_agent_status(self, session_id: str) -> Dict[str, Any]:
        """Get current agent status and context"""
        context = self._get_conversation_context(session_id)
        
        return {
            "state": self.state.value,
            "session_id": session_id,
            "user_preferences": context.user_preferences,
            "cart_items": len(context.current_cart),
            "dietary_restrictions": context.dietary_restrictions,
            "last_intent": context.last_intent,
            "context_summary": context.context_summary[:200] + "..." if len(context.context_summary) > 200 else context.context_summary
        }
    
    def clear_session_context(self, session_id: str):
        """Clear cached context for a session"""
        if session_id in self.context_cache:
            del self.context_cache[session_id]
        self.memory.clear()

# Global agent instance
intelligent_agent = IntelligentFoodAgent()

async def process_user_message(user_input: str, session_id: str) -> str:
    """Main entry point for processing user messages"""
    try:
        response = await intelligent_agent.process_message(user_input, session_id)
        return response
    except Exception as e:
        print(f"Message processing error: {e}")
        return "I'm having some technical difficulties right now. Please try again in a moment, or let me know if you need immediate assistance!"

def get_agent_status(session_id: str) -> Dict[str, Any]:
    """Get agent status for debugging/monitoring"""
    return intelligent_agent.get_agent_status(session_id)

def reset_agent_session(session_id: str):
    """Reset agent session context"""
    intelligent_agent.clear_session_context(session_id)

# Synchronous wrapper for compatibility
def process_query(query: str, session_id: str) -> str:
    """Synchronous wrapper for the async agent"""
    try:
        # Run the async function
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            response = loop.run_until_complete(process_user_message(query, session_id))
            return response
        finally:
            loop.close()
    except Exception as e:
        print(f"Sync wrapper error: {e}")
        return "I'm experiencing some technical issues. Please try again or contact support if the problem persists."