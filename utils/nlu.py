import os, json, traceback
from typing import Any, Dict, List
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
    filters: Dict[str, Any] = Field(default_factory=dict)  # e.g., {"diet":"veg"}
    delivery_mode: str | None = None
    customer_name: str | None = None
    customer_phone: str | None = None
    customer_address: str | None = None
    promo_code: str | None = None
    confidence: float = 1.0

parser = JsonOutputParser(pydantic_object=NLUResult)

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
prompt = PromptTemplate(template=PROMPT, input_variables=["utterance"], partial_variables={"format_instructions": parser.get_format_instructions()})

FORCE_PROMPT = """
You MUST return JSON as described. If uncertain, guess the most likely intent and content and set confidence accordingly.
Return ONLY JSON.
{format_instructions}

User: {utterance}
"""
force_prompt = PromptTemplate(template=FORCE_PROMPT, input_variables=["utterance"], partial_variables={"format_instructions": parser.get_format_instructions()})

def _make_llm(temp: float = 0.5):
    key = os.getenv("GROQ_API_KEY")
    return ChatGroq(model="openai/gpt-oss-20b", groq_api_key=key, temperature=temp)

def _to_dict_safe(raw: Any) -> Dict[str, Any]:
    try:
        if isinstance(raw, dict):
            return raw
        return raw.dict()  # pydantic model
    except Exception:
        try:
            return json.loads(str(raw))
        except Exception:
            traceback.print_exc()
            return {"intent":"help","items":[],"filters":{},"confidence":0.0}

def nlu_extract(llm: ChatGroq | None, utterance: str, min_confidence: float = 0.45) -> Dict[str, Any]:
    if llm is None:
        llm = _make_llm(0.0)
    # first pass (stable)
    try:
        chain = prompt | llm | parser
        raw = chain.invoke({"utterance": utterance})
        parsed = _to_dict_safe(raw)
    except Exception as e:
        print("NLU primary failed:", e)
        traceback.print_exc()
        parsed = {"intent":"help","items":[],"filters":{},"confidence":0.0}

    parsed = dict(parsed)
    if "confidence" not in parsed:
        parsed["confidence"] = 1.0 if parsed.get("intent") and parsed.get("intent")!="help" else 0.0

    try:
        parsed["confidence"] = float(parsed.get("confidence", 0.0))
    except Exception:
        parsed["confidence"] = 0.0

    # if low confidence, forced second pass to guess
    if parsed["confidence"] < min_confidence:
        try:
            llm2 = _make_llm(0.25)
            chain2 = force_prompt | llm2 | parser
            raw2 = chain2.invoke({"utterance": utterance})
            parsed2 = _to_dict_safe(raw2)
            parsed2 = dict(parsed2)
            parsed2["confidence"] = float(parsed2.get("confidence", 0.0))
            # choose higher-confidence parse
            if parsed2["confidence"] >= parsed["confidence"]:
                parsed = parsed2
            else:
                # keep parsed but ensure minimally usable
                parsed["confidence"] = max(parsed["confidence"], 0.1)
        except Exception as e:
            print("NLU forced failed:", e)
            traceback.print_exc()
            parsed["confidence"] = max(parsed.get("confidence",0.0), 0.0)

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
    # sanitize items
    items_raw = parsed.get("items") or []
    sanitized = []
    for it in items_raw:
        try:
            if isinstance(it, dict):
                name = str(it.get("name","")).strip()
                qty = int(it.get("quantity", 1) or 1)
                mods = it.get("modifiers") or []
            else:
                name = str(getattr(it,"name",str(it))).strip()
                qty = int(getattr(it,"quantity",1) or 1)
                mods = getattr(it,"modifiers",[]) or []
            if name:
                sanitized.append({"name": name, "quantity": max(1, qty), "modifiers": mods})
        except Exception:
            continue
    out["items"] = sanitized[:20]
    out["filters"] = out["filters"] if isinstance(out["filters"], dict) else {}
    out["confidence"] = max(0.0, min(1.0, out["confidence"]))
    return out
