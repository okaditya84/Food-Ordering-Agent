# utils/menu_matcher.py
import json, re
from typing import List, Dict, Any, Tuple
from rapidfuzz import process, fuzz

with open("menu.json", "r", encoding="utf-8") as f:
    MENU = json.load(f)

CANONICAL_ITEMS = []
MODIFIERS = {}
for cat in MENU:
    if cat.get("category","").lower() == "modifiers":
        for m in cat["items"]:
            MODIFIERS[m["name"].lower()] = float(m.get("price_delta", 0.0))
        continue
    for it in cat.get("items", []):
        CANONICAL_ITEMS.append({
            "name": it.get("name"),
            "price": float(it.get("price", 0.0)),
            "description": it.get("description",""),
            "category": cat.get("category",""),
            "toppings": it.get("toppings", []),
            "veg": bool(it.get("veg", False)),
            "tags": it.get("tags", [])
        })

NAMES = [c["name"] for c in CANONICAL_ITEMS if c["name"]]

def extract_modifiers_and_base(raw_text: str) -> Tuple[str, List[str]]:
    text = (raw_text or "").lower()
    found = []
    for m in MODIFIERS.keys():
        if re.search(r"\b"+re.escape(m)+r"\b", text):
            found.append(m)
            text = re.sub(r"\b"+re.escape(m)+r"\b", " ", text)
    text = re.sub(r"\b(and|with|but|make it|please|without)\b", " ", text)
    text = re.sub(r"[^\w\s]", " ", text)
    base = " ".join(text.split()).strip()
    return (base if base else raw_text, found)

def canonicalize_name(raw: str) -> str:
    base, _ = extract_modifiers_and_base(raw)
    match = process.extractOne(base, NAMES, scorer=fuzz.WRatio)
    if not match:
        return raw
    name, score, idx = match
    return CANONICAL_ITEMS[idx]["name"]

def price_of(name: str) -> float:
    n = (name or "").strip().lower()
    for c in CANONICAL_ITEMS:
        if c["name"] and c["name"].lower()==n:
            return float(c.get("price", 0.0))
    # fuzzy fallback
    match = process.extractOne(name, NAMES, scorer=fuzz.WRatio)
    if match:
        _,__, idx = match
        return CANONICAL_ITEMS[idx]["price"]
    return 0.0

def get_item_by_name(name: str) -> Dict[str,Any]:
    n = (name or "").strip().lower()
    for c in CANONICAL_ITEMS:
        if c["name"] and c["name"].lower()==n:
            return c
    match = process.extractOne(name, NAMES, scorer=fuzz.WRatio)
    if match:
        _,__, idx = match
        return CANONICAL_ITEMS[idx]
    return {}

class NormalizedItem:
    def __init__(self, name: str, quantity: int, base_price: float, modifiers: List[str], veg: bool=False, toppings: List[str]=[]):
        self.name = name
        self.quantity = int(quantity)
        self.base_price = float(base_price)
        self.modifiers = modifiers or []
        self.veg = veg
        self.toppings = toppings or []

    def modifier_delta(self) -> float:
        return sum(MODIFIERS.get(m, 0.0) for m in self.modifiers)

    def unit_price(self) -> float:
        return round(self.base_price + self.modifier_delta(), 2)

    def total_price(self) -> float:
        return round(self.unit_price() * self.quantity, 2)

    def to_dict(self) -> Dict[str,Any]:
        return {
            "name": self.name,
            "quantity": self.quantity,
            "base_price": self.base_price,
            "unit_price": self.unit_price(),
            "total_price": self.total_price(),
            "modifiers": self.modifiers,
            "veg": self.veg,
            "toppings": self.toppings
        }

def normalize_items_against_menu(items: List[Dict[str,Any]]) -> List[NormalizedItem]:
    out = []
    for it in items:
        if isinstance(it, dict):
            raw_name = it.get("name", "")
            qty = int(it.get("quantity", 1))
            mods = it.get("modifiers", []) or []
        else:
            raw_name = getattr(it, "name", str(it))
            qty = getattr(it, "quantity", 1)
            mods = getattr(it, "modifiers", []) or []
        base, found_mods = extract_modifiers_and_base(raw_name)
        # merge explicit mods from both places
        merged_mods = list(set(found_mods + (mods or [])))
        match = process.extractOne(base, NAMES, scorer=fuzz.WRatio)
        if match:
            name, score, idx = match
            meta = CANONICAL_ITEMS[idx]
            out.append(NormalizedItem(meta["name"], qty, meta["price"], merged_mods, meta.get("veg", False), meta.get("toppings", [])))
        else:
            out.append(NormalizedItem(raw_name, qty, 0.0, merged_mods, False, []))
    return out
