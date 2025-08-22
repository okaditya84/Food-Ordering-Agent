# utils/nlu.py  -- REPLACE your existing file with this
import os
import json
import traceback
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

class ItemSchema(BaseModel):
    name: str
    quantity: int = 1
    modifiers: List[str] = []

class NLUResult(BaseModel):
    intent: str
    items: List[ItemSchema] = Field(default_factory=list)
    filters: Dict[str, Any] = Field(default_factory=dict)
    delivery_mode: Optional[str] = None
    customer_name: Optional[str] = None
    customer_phone: Optional[str] = None
    customer_address: Optional[str] = None
    promo_code: Optional[str] = None
    confidence: float = 0.7

parser = JsonOutputParser(pydantic_object=NLUResult)

# Raw prompt (keep instructions). We'll inject escaped format_instructions below.
PROMPT = """
Parse the user's utterance into strict JSON. The intent must be one of:
- menu.show, menu.search, menu.info, menu.filter, order.add, order.replace, order.remove, order.confirm, order.cancel, history.show, smalltalk, help

Extract:
- items: [{name, quantity, modifiers}] (if any)
- filters: e.g., {"diet":"veg"} or {"tag":"pizza"}
- delivery_mode, customer_name, customer_phone, customer_address, promo_code
- confidence: 0.0 - 1.0 indicating parser confidence

Return ONLY JSON with these fields (valid JSON conforming to schema).
{format_instructions}
User: {utterance}
"""

FORCE_PROMPT = """
You MUST return JSON as described. If uncertain, guess the most likely intent and content and set confidence accordingly.
Return ONLY JSON.
{format_instructions}
User: {utterance}
"""

def _escape_braces(s: str) -> str:
    """
    Escape single braces so PromptTemplate won't interpret them as template variables.
    This preserves the literal formatting instructions from JsonOutputParser.
    """
    if not isinstance(s, str):
        return s
    return s.replace("{", "{{").replace("}", "}}")

# build prompt templates with escaped format_instructions to avoid KeyError
escaped_instructions = _escape_braces(parser.get_format_instructions())

prompt = PromptTemplate(
    template=PROMPT,
    input_variables=["utterance"],
    partial_variables={"format_instructions": escaped_instructions}
)

force_prompt = PromptTemplate(
    template=FORCE_PROMPT,
    input_variables=["utterance"],
    partial_variables={"format_instructions": escaped_instructions}
)

def _make_llm(temp: float = 0.5) -> ChatGroq:
    """Creates a ChatGroq instance with a specified temperature using env var."""
    key = os.getenv("GROQ_API_KEY")
    if not key:
        # fallback: allow tool invocation but do not raise here; upstream can handle None
        raise ValueError("GROQ_API_KEY not found in environment variables.")
    return ChatGroq(model="openai/gpt-oss-20b", groq_api_key=key, temperature=temp)

def _to_dict_safe(raw: Any) -> Dict[str, Any]:
    """Safely converts an object (Pydantic model or dict) to a dictionary."""
    try:
        if isinstance(raw, dict):
            return raw
        # pydantic v2
        return raw.model_dump()
    except Exception:
        try:
            return json.loads(str(raw))
        except Exception:
            traceback.print_exc()
            return {"intent": "help", "items": [], "filters": {}, "confidence": 0.0}

def _rule_based_fallback(utterance: str) -> Dict[str, Any]:
    """
    Simple rule-based fallback to guarantee behavior for common utterances
    (helps when LLM is unavailable or returns an error).
    """
    text = (utterance or "").lower()
    out = {"intent": "help", "items": [], "filters": {}, "confidence": 0.0}
    if any(w in text for w in ["menu", "show menu", "show me the menu", "show me the restaurant menu", "see menu"]):
        out["intent"] = "menu.show"
        out["confidence"] = 1.0
        return out
    if any(w in text for w in ["history", "order history", "my orders"]):
        out["intent"] = "history.show"
        out["confidence"] = 1.0
        return out
    if any(w in text for w in ["cancel order", "cancel my order", "cancel"]):
        out["intent"] = "order.cancel"
        out["confidence"] = 0.8
        return out
    if any(w in text for w in ["add", "remove", "order", "confirm"]):
        # weak guess
        if "confirm" in text:
            out["intent"] = "order.confirm"
        elif "remove" in text or "delete" in text:
            out["intent"] = "order.remove"
        else:
            out["intent"] = "order.add"
        out["confidence"] = 0.4
        return out
    # fallback smalltalk/help
    out["intent"] = "smalltalk"
    out["confidence"] = 0.2
    return out

def nlu_extract(llm: Optional[ChatGroq], utterance: str, min_confidence: float = 0.45) -> Dict[str, Any]:
    """
    Extracts NLU information from a user utterance using a two-pass LLM approach.
    Falls back to a rule-based parser if the LLM fails.
    """
    # quick sanity
    if not utterance:
        return {"intent": "help", "items": [], "filters": {}, "confidence": 0.0}

    if llm is None:
        # try to create a low-temperature LLM; if unavailable, we'll fallback
        try:
            llm = _make_llm(0.0)
        except Exception as e:
            # LLM unavailable: return rule-based fallback
            print("NLU: LLM unavailable, using rule-based fallback.", e)
            return _rule_based_fallback(utterance)

    # First pass (stable)
    parsed = None
    try:
        chain = prompt | llm | parser
        raw = chain.invoke({"utterance": utterance})
        parsed = _to_dict_safe(raw)
    except Exception as e:
        print("NLU primary failed:", e)
        traceback.print_exc()
        # fallback to a quick rule-based parse so we can still respond
        return _rule_based_fallback(utterance)

    parsed = dict(parsed or {})

    if "confidence" not in parsed or parsed["confidence"] is None:
        parsed["confidence"] = 1.0 if parsed.get("intent") and parsed.get("intent") != "help" else 0.0

    try:
        parsed["confidence"] = float(parsed["confidence"])
    except (ValueError, TypeError):
        parsed["confidence"] = 0.0

    # If low confidence, forced second pass is made to guess
    if parsed.get("confidence", 0.0) < min_confidence:
        try:
            llm2 = _make_llm(0.25)
            chain2 = force_prompt | llm2 | parser
            raw2 = chain2.invoke({"utterance": utterance})
            parsed2 = _to_dict_safe(raw2)
            parsed2 = dict(parsed2 or {})
            parsed2["confidence"] = float(parsed2.get("confidence", 0.0))
            if parsed2["confidence"] >= parsed["confidence"]:
                parsed = parsed2
            else:
                parsed["confidence"] = max(parsed["confidence"], 0.1)
        except Exception as e:
            print("NLU forced failed:", e)
            traceback.print_exc()
            parsed["confidence"] = max(parsed.get("confidence", 0.0), 0.0)

    # Normalize result keys and types, sanitize items
    out = {
        "intent": str(parsed.get("intent") or "help"),
        "items": [],
        "filters": parsed.get("filters") or {},
        "delivery_mode": parsed.get("delivery_mode"),
        "customer_name": parsed.get("customer_name"),
        "customer_phone": parsed.get("customer_phone"),
        "customer_address": parsed.get("customer_address"),
        "promo_code": parsed.get("promo_code"),
        "confidence": float(parsed.get("confidence", 0.0) or 0.0)
    }

    # Sanitize and populate items
    items_raw = parsed.get("items") or []
    sanitized = []
    for it in items_raw:
        try:
            if isinstance(it, dict):
                name = str(it.get("name", "")).strip()
                qty = int(it.get("quantity", 1) or 1)
                mods = it.get("modifiers", []) or []
            else:
                # Fallback for non-dict items
                name = str(getattr(it, "name", str(it))).strip()
                qty = int(getattr(it, "quantity", 1) or 1)
                mods = getattr(it, "modifiers", []) or []
            if name:
                sanitized.append({"name": name, "quantity": max(1, qty), "modifiers": mods})
        except Exception:
            continue

    out["items"] = sanitized[:20]
    out["filters"] = out["filters"] if isinstance(out["filters"], dict) else {}
    out["confidence"] = max(0.0, min(1.0, out["confidence"]))

    return out
