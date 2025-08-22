import os
import json
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_groq import ChatGroq
from sentence_transformers import CrossEncoder
import spacy

from config import get_config, GROQ_API_KEY, EMBEDDING_MODEL, RERANKER_MODEL
from database import log_conversation

@dataclass
class RetrievalResult:
    content: str
    score: float
    metadata: Dict[str, Any]
    source: str
    relevance_explanation: Optional[str] = None

class QueryExpansionEngine:
    """Expands user queries with synonyms and related terms"""
    
    def __init__(self):
        self.llm = self._create_llm()
        self.nlp = self._load_spacy()
        self.food_synonyms = self._load_food_synonyms()
        
    def _create_llm(self) -> Optional[ChatGroq]:
        try:
            return ChatGroq(
                model="llama3-70b-8192",
                groq_api_key=GROQ_API_KEY,
                temperature=0.3,
                max_tokens=512
            )
        except Exception as e:
            print(f"Failed to create LLM for query expansion: {e}")
            return None
    
    def _load_spacy(self):
        try:
            return spacy.load("en_core_web_sm")
        except OSError:
            return None
    
    def _load_food_synonyms(self) -> Dict[str, List[str]]:
        """Load food-related synonyms and related terms"""
        return {
            "pizza": ["pie", "flatbread", "margherita", "pepperoni", "italian"],
            "burger": ["sandwich", "patty", "beef", "chicken", "veggie"],
            "pasta": ["noodles", "spaghetti", "fettuccine", "italian", "alfredo"],
            "salad": ["greens", "vegetables", "healthy", "fresh", "lettuce"],
            "drink": ["beverage", "liquid", "soda", "juice", "water"],
            "dessert": ["sweet", "cake", "ice cream", "chocolate", "sugar"],
            "spicy": ["hot", "chili", "pepper", "heat", "fire"],
            "mild": ["gentle", "light", "soft", "not spicy"],
            "vegetarian": ["veggie", "plant-based", "no meat", "veg"],
            "vegan": ["plant-based", "no dairy", "no animal products"],
            "healthy": ["nutritious", "low calorie", "diet", "fitness", "wellness"],
            "cheap": ["affordable", "budget", "inexpensive", "low cost"],
            "expensive": ["premium", "high-end", "costly", "luxury"]
        }
    
    def expand_with_synonyms(self, query: str) -> List[str]:
        """Expand query with synonyms"""
        expanded_terms = [query]
        query_lower = query.lower()
        
        for term, synonyms in self.food_synonyms.items():
            if term in query_lower:
                for synonym in synonyms:
                    expanded_query = query_lower.replace(term, synonym)
                    if expanded_query != query_lower:
                        expanded_terms.append(expanded_query)
        
        return list(set(expanded_terms))
    
    def expand_with_llm(self, query: str, context: str = "") -> List[str]:
        """Expand query using LLM understanding"""
        if not self.llm:
            return [query]
        
        try:
            prompt = f"""
            Given this food ordering query: "{query}"
            Context: {context}
            
            Generate 3-5 alternative ways to express the same intent, focusing on:
            1. Different food terminology
            2. Various ways to express preferences
            3. Related menu categories
            4. Common customer language variations
            
            Return only the alternative queries, one per line, without numbering.
            """
            
            response = self.llm.invoke(prompt)
            content = response.content if hasattr(response, 'content') else str(response)
            
            expanded_queries = [line.strip() for line in content.split('\n') if line.strip()]
            return [query] + expanded_queries[:4]  # Original + up to 4 expansions
            
        except Exception as e:
            print(f"LLM query expansion failed: {e}")
            return [query]
    
    def expand_with_entities(self, query: str) -> List[str]:
        """Expand query based on named entities"""
        if not self.nlp:
            return [query]
        
        doc = self.nlp(query)
        expanded_queries = [query]
        
        # Extract entities and create variations
        entities = [(ent.text, ent.label_) for ent in doc.ents]
        
        for entity_text, entity_label in entities:
            if entity_label in ["FOOD", "PRODUCT", "ORG"]:
                # Create queries without the specific entity (more general)
                general_query = query.replace(entity_text, "").strip()
                if general_query and general_query != query:
                    expanded_queries.append(general_query)
        
        return list(set(expanded_queries))
    
    def expand_query(self, query: str, method: str = "all", context: str = "") -> List[str]:
        """Main query expansion method"""
        if method == "synonyms":
            return self.expand_with_synonyms(query)
        elif method == "llm":
            return self.expand_with_llm(query, context)
        elif method == "entities":
            return self.expand_with_entities(query)
        else:  # "all"
            synonym_expansions = self.expand_with_synonyms(query)
            entity_expansions = self.expand_with_entities(query)
            
            # Combine and deduplicate
            all_expansions = list(set(synonym_expansions + entity_expansions))
            
            # Limit to prevent too many queries
            return all_expansions[:6]

class IntelligentReranker:
    """Reranks retrieved documents based on relevance and context"""
    
    def __init__(self):
        self.cross_encoder = self._load_cross_encoder()
        self.llm = self._create_llm()
    
    def _load_cross_encoder(self):
        """Load cross-encoder model for reranking"""
        try:
            return CrossEncoder(RERANKER_MODEL)
        except Exception as e:
            print(f"Failed to load cross-encoder: {e}")
            return None
    
    def _create_llm(self):
        try:
            return ChatGroq(
                model="llama3-70b-8192",
                groq_api_key=GROQ_API_KEY,
                temperature=0.1,
                max_tokens=1024
            )
        except Exception:
            return None
    
    def rerank_with_cross_encoder(self, query: str, documents: List[Document]) -> List[Tuple[Document, float]]:
        """Rerank documents using cross-encoder"""
        if not self.cross_encoder or not documents:
            return [(doc, 0.5) for doc in documents]
        
        try:
            # Prepare query-document pairs
            pairs = [(query, doc.page_content) for doc in documents]
            
            # Get relevance scores
            scores = self.cross_encoder.predict(pairs)
            
            # Combine documents with scores
            doc_scores = list(zip(documents, scores))
            
            # Sort by score (descending)
            doc_scores.sort(key=lambda x: x[1], reverse=True)
            
            return doc_scores
            
        except Exception as e:
            print(f"Cross-encoder reranking failed: {e}")
            return [(doc, 0.5) for doc in documents]
    
    def rerank_with_llm(self, query: str, documents: List[Document], context: str = "") -> List[Tuple[Document, float]]:
        """Rerank documents using LLM understanding"""
        if not self.llm or not documents:
            return [(doc, 0.5) for doc in documents]
        
        try:
            # Prepare document summaries
            doc_summaries = []
            for i, doc in enumerate(documents):
                summary = doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
                doc_summaries.append(f"{i}: {summary}")
            
            prompt = f"""
            Query: "{query}"
            Context: {context}
            
            Documents to rank:
            {chr(10).join(doc_summaries)}
            
            Rank these documents by relevance to the query (0-10 scale).
            Consider:
            1. Direct relevance to the query
            2. Completeness of information
            3. Usefulness for food ordering
            
            Return only the rankings as: "0:score,1:score,2:score" etc.
            """
            
            response = self.llm.invoke(prompt)
            content = response.content if hasattr(response, 'content') else str(response)
            
            # Parse rankings
            rankings = {}
            for item in content.split(','):
                if ':' in item:
                    try:
                        idx, score = item.strip().split(':')
                        rankings[int(idx)] = float(score) / 10.0  # Normalize to 0-1
                    except (ValueError, IndexError):
                        continue
            
            # Apply rankings
            doc_scores = []
            for i, doc in enumerate(documents):
                score = rankings.get(i, 0.5)  # Default score if not found
                doc_scores.append((doc, score))
            
            # Sort by score
            doc_scores.sort(key=lambda x: x[1], reverse=True)
            return doc_scores
            
        except Exception as e:
            print(f"LLM reranking failed: {e}")
            return [(doc, 0.5) for doc in documents]
    
    def rerank_documents(self, query: str, documents: List[Document], method: str = "cross_encoder", context: str = "") -> List[Tuple[Document, float]]:
        """Main reranking method"""
        if method == "cross_encoder":
            return self.rerank_with_cross_encoder(query, documents)
        elif method == "llm":
            return self.rerank_with_llm(query, documents, context)
        else:  # hybrid
            # Use cross-encoder first, then LLM for top results
            ce_results = self.rerank_with_cross_encoder(query, documents)
            top_docs = [doc for doc, score in ce_results[:5]]  # Top 5 from cross-encoder
            
            if len(top_docs) > 1:
                llm_results = self.rerank_with_llm(query, top_docs, context)
                # Combine remaining documents
                remaining_docs = [(doc, score) for doc, score in ce_results[5:]]
                return llm_results + remaining_docs
            else:
                return ce_results

class IntelligentKnowledgeRetrieval:
    """Main RAG system with advanced capabilities"""
    
    def __init__(self, menu_file: str = "data/restaurant_menu.json"):
        self.config = get_config("rag")
        self.menu_file = menu_file
        self.embeddings = self._create_embeddings()
        self.vectorstore = None
        self.query_expander = QueryExpansionEngine()
        self.reranker = IntelligentReranker()
        self.llm = self._create_llm()
        
        # Load and index menu
        self._load_and_index_menu()
    
    def _create_embeddings(self):
        """Create embedding model"""
        return HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
    
    def _create_llm(self):
        try:
            return ChatGroq(
                model="llama3-70b-8192",
                groq_api_key=GROQ_API_KEY,
                temperature=0.2,
                max_tokens=1024
            )
        except Exception:
            return None
    
    def _load_and_index_menu(self):
        """Load menu data and create vector index"""
        try:
            # Try enhanced menu first, fallback to original
            menu_files = ["menu.json"]
            menu_data = None
            
            for menu_file in menu_files:
                if os.path.exists(menu_file):
                    with open(menu_file, 'r', encoding='utf-8') as f:
                        menu_data = json.load(f)
                    break
            
            if not menu_data:
                print("No menu file found")
                return
            
            # Create documents
            documents = []
            for category in menu_data:
                category_name = category.get("category", "")
                
                for item in category.get("items", []):
                    # Create rich document content
                    content_parts = [
                        f"Category: {category_name}",
                        f"Name: {item.get('name', '')}",
                        f"Price: ${item.get('price', 0):.2f}",
                        f"Description: {item.get('description', '')}",
                    ]
                    
                    # Add nutritional information
                    nutrition = item.get('nutrition', {})
                    if nutrition:
                        content_parts.append(f"Calories: {nutrition.get('calories', 'N/A')}")
                        content_parts.append(f"Protein: {nutrition.get('protein', 'N/A')}g")
                        content_parts.append(f"Carbs: {nutrition.get('carbs', 'N/A')}g")
                        content_parts.append(f"Fat: {nutrition.get('fat', 'N/A')}g")
                    
                    # Add dietary information
                    dietary_tags = item.get('dietary_tags', [])
                    if dietary_tags:
                        content_parts.append(f"Dietary: {', '.join(dietary_tags)}")
                    
                    allergens = item.get('allergens', [])
                    if allergens:
                        content_parts.append(f"Allergens: {', '.join(allergens)}")
                    
                    ingredients = item.get('ingredients', [])
                    if ingredients:
                        content_parts.append(f"Ingredients: {', '.join(ingredients)}")
                    
                    # Additional metadata
                    spice_level = item.get('spice_level', 0)
                    if spice_level > 0:
                        content_parts.append(f"Spice Level: {spice_level}/5")
                    
                    prep_time = item.get('prep_time', 0)
                    if prep_time > 0:
                        content_parts.append(f"Prep Time: {prep_time} minutes")
                    
                    content = "\n".join(content_parts)
                    
                    # Create document
                    doc = Document(
                        page_content=content,
                        metadata={
                            "category": category_name,
                            "name": item.get('name', ''),
                            "price": item.get('price', 0),
                            "dietary_tags": dietary_tags,
                            "allergens": allergens,
                            "nutrition": nutrition,
                            "spice_level": spice_level,
                            "prep_time": prep_time,
                            "popularity_score": item.get('popularity_score', 0.5),
                            "customizable": item.get('customizable', False)
                        }
                    )
                    documents.append(doc)
            
            # Split documents if needed
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.config.get("chunk_size", 512),
                chunk_overlap=self.config.get("chunk_overlap", 50)
            )
            split_docs = text_splitter.split_documents(documents)
            
            # Create vector store
            self.vectorstore = FAISS.from_documents(split_docs, self.embeddings)
            
            print(f"Indexed {len(split_docs)} menu documents")
            
        except Exception as e:
            print(f"Failed to load and index menu: {e}")
            self.vectorstore = None
    
    def retrieve_documents(self, query: str, k: int = None) -> List[Document]:
        """Retrieve documents using similarity search"""
        if not self.vectorstore:
            return []
        
        k = k or self.config.get("top_k_retrieval", 15)
        
        try:
            # Basic similarity search
            docs = self.vectorstore.similarity_search(query, k=k)
            return docs
        except Exception as e:
            print(f"Document retrieval failed: {e}")
            return []
    
    def retrieve_with_expansion(self, query: str, context: str = "", k: int = None) -> List[Document]:
        """Retrieve documents with query expansion"""
        if not self.config.get("query_expansion", True):
            return self.retrieve_documents(query, k)
        
        # Expand query
        expanded_queries = self.query_expander.expand_query(query, context=context)
        
        # Retrieve for each expanded query
        all_docs = []
        k_per_query = max(1, (k or self.config.get("top_k_retrieval", 15)) // len(expanded_queries))
        
        for exp_query in expanded_queries:
            docs = self.retrieve_documents(exp_query, k=k_per_query)
            all_docs.extend(docs)
        
        # Remove duplicates based on content
        unique_docs = []
        seen_content = set()
        
        for doc in all_docs:
            content_hash = hash(doc.page_content)
            if content_hash not in seen_content:
                seen_content.add(content_hash)
                unique_docs.append(doc)
        
        return unique_docs[:k or self.config.get("top_k_retrieval", 15)]
    
    def filter_by_dietary_preferences(self, documents: List[Document], dietary_prefs: List[str]) -> List[Document]:
        """Filter documents based on dietary preferences"""
        if not dietary_prefs:
            return documents
        
        filtered_docs = []
        for doc in documents:
            doc_dietary_tags = doc.metadata.get("dietary_tags", [])
            doc_allergens = doc.metadata.get("allergens", [])
            
            # Check if document matches dietary preferences
            matches_prefs = True
            for pref in dietary_prefs:
                if pref.lower() in ["vegetarian", "vegan", "gluten_free", "dairy_free", "keto"]:
                    if pref.lower() not in [tag.lower() for tag in doc_dietary_tags]:
                        matches_prefs = False
                        break
                elif pref.lower().startswith("no_"):
                    # Handle "no_dairy", "no_gluten" etc.
                    allergen = pref[3:]  # Remove "no_"
                    if allergen in [a.lower() for a in doc_allergens]:
                        matches_prefs = False
                        break
            
            if matches_prefs:
                filtered_docs.append(doc)
        
        return filtered_docs
    
    def generate_response(self, query: str, documents: List[Document], context: str = "") -> str:
        """Generate response using retrieved documents"""
        if not self.llm or not documents:
            return "I couldn't find relevant information to answer your question."
        
        # Prepare document content
        doc_content = "\n\n".join([
            f"Item {i+1}:\n{doc.page_content}" 
            for i, doc in enumerate(documents[:5])  # Limit to top 5
        ])
        
        prompt = f"""
        You are a helpful food ordering assistant. Use the following menu information to answer the user's question.
        
        User Question: {query}
        Context: {context}
        
        Menu Information:
        {doc_content}
        
        Instructions:
        1. Provide accurate information based only on the menu data
        2. Be helpful and conversational
        3. Include prices, dietary information, and descriptions when relevant
        4. If asking about availability, mention what's available
        5. Suggest alternatives if the exact request isn't available
        6. Format the response in a user-friendly way
        
        Response:
        """
        
        try:
            response = self.llm.invoke(prompt)
            return response.content if hasattr(response, 'content') else str(response)
        except Exception as e:
            print(f"Response generation failed: {e}")
            return "I'm having trouble generating a response right now. Please try again."
    
    def search(self, 
               query: str, 
               session_id: str = None,
               context: str = "", 
               dietary_prefs: List[str] = None,
               use_reranking: bool = None,
               k: int = None) -> Dict[str, Any]:
        """Main search method with all advanced features"""
        
        use_reranking = use_reranking if use_reranking is not None else self.config.get("use_reranking", True)
        k = k or self.config.get("rerank_top_k", 8)
        
        # Step 1: Retrieve documents with expansion
        documents = self.retrieve_with_expansion(query, context, k=self.config.get("top_k_retrieval", 15))
        
        if not documents:
            return {
                "response": "I couldn't find any menu items matching your request. Could you try rephrasing your question?",
                "documents": [],
                "metadata": {"total_docs": 0, "method": "no_results"}
            }
        
        # Step 2: Filter by dietary preferences
        if dietary_prefs:
            documents = self.filter_by_dietary_preferences(documents, dietary_prefs)
        
        # Step 3: Rerank documents
        if use_reranking and len(documents) > 1:
            doc_scores = self.reranker.rerank_documents(query, documents, method="hybrid", context=context)
            documents = [doc for doc, score in doc_scores[:k]]
        else:
            documents = documents[:k]
        
        # Step 4: Generate response
        response = self.generate_response(query, documents, context)
        
        # Step 5: Log the interaction
        if session_id:
            log_conversation(session_id, "system", f"RAG search: {query}", {
                "documents_found": len(documents),
                "dietary_filters": dietary_prefs or [],
                "used_reranking": use_reranking
            })
        
        return {
            "response": response,
            "documents": [
                {
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "relevance_score": 0.8  # Placeholder score
                }
                for doc in documents
            ],
            "metadata": {
                "total_docs": len(documents),
                "method": "advanced_rag",
                "used_expansion": self.config.get("query_expansion", True),
                "used_reranking": use_reranking,
                "dietary_filters": dietary_prefs or []
            }
        }

# Global RAG instance
knowledge_retrieval = IntelligentKnowledgeRetrieval()

def search_menu(query: str, 
                session_id: str = None,
                context: str = "",
                dietary_prefs: List[str] = None,
                k: int = 8) -> str:
    """Main entry point for menu search"""
    try:
        result = knowledge_retrieval.search(
            query=query,
            session_id=session_id,
            context=context,
            dietary_prefs=dietary_prefs,
            k=k
        )
        return result["response"]
    
    except Exception as e:
        print(f"Menu search error: {e}")
        return "I'm having trouble searching the menu right now. Please try again in a moment."

def get_menu_recommendations(dietary_prefs: List[str] = None, 
                           price_range: Tuple[float, float] = None,
                           session_id: str = None) -> str:
    """Get personalized menu recommendations"""
    query = "recommend popular and highly rated items"
    
    if dietary_prefs:
        pref_text = ", ".join(dietary_prefs)
        query += f" that are {pref_text}"
    
    if price_range:
        min_price, max_price = price_range
        query += f" between ${min_price:.2f} and ${max_price:.2f}"
    
    return search_menu(query, session_id=session_id, dietary_prefs=dietary_prefs)