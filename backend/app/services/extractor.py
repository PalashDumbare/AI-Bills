import re
from datetime import date, datetime
from dateutil import parser as date_parser


KNOWN_BRANDS = [
    "LG", "Samsung", "Sony", "Whirlpool", "IFB", "Bosch", "Philips",
    "Panasonic", "Haier", "Voltas", "Daikin", "Blue Star", "Hitachi",
    "Toshiba", "Oppo", "Vivo", "OnePlus", "Xiaomi", "Apple", "Dell",
    "HP", "Lenovo", "Asus", "Acer", "Canon", "Nikon", "Westinghouse",
]

BILL_TYPES = ["electricity", "gas", "water", "internet", "mobile", "broadband"]


async def extract_structured_data_auto(text: str) -> dict | None:
    """Try LLM extraction first, fallback to regex."""
    try:
        from .llm_client import extract_with_llm
        result = await extract_with_llm(text)
        if result:
            return result
    except Exception:
        pass

    return extract_structured_data(text)


def extract_structured_data(text: str) -> dict | None:
    """Extract structured data from text using rule-based parsing."""
    lines = [line.strip() for line in text.strip().split("\n") if line.strip()]
    if not lines:
        return None

    result = {}

    result["brand"] = _extract_brand(lines)
    result["product"] = _extract_product(lines)
    result["model"] = _extract_model(text)
    result["purchase_date"] = _extract_date(text)
    result["amount"] = _extract_amount(text)
    result["warranty_months"] = _extract_warranty_months(text)
    result["bill_type"] = _extract_bill_type(text)
    result["provider"] = _extract_provider(lines)

    has_appliance_fields = any([result["brand"], result["product"], result["model"]])
    has_bill_fields = any([result["bill_type"], result["provider"]])

    if not has_appliance_fields and not has_bill_fields:
        return None

    if has_appliance_fields:
        result["document_type"] = "appliance_invoice"
        if result["warranty_months"] and result["purchase_date"]:
            from dateutil.relativedelta import relativedelta
            result["warranty_expiry"] = (
                result["purchase_date"] + relativedelta(months=result["warranty_months"])
            )
    else:
        result["document_type"] = "bill"

    return result


def _extract_brand(lines: list[str]) -> str | None:
    for line in lines[:3]:
        for brand in KNOWN_BRANDS:
            if brand.lower() in line.lower():
                return brand
    return None


def _extract_product(lines: list[str]) -> str | None:
    product_keywords = [
        "washing machine", "refrigerator", "fridge", "tv", "television",
        "laptop", "ac", "air conditioner", "phone", "mobile", "tablet",
        "microwave", "oven", "heater", "fan", "water purifier",
    ]
    for line in lines[:5]:
        for keyword in product_keywords:
            if keyword.lower() in line.lower():
                return keyword.title()
    return None


def _extract_model(text: str) -> str | None:
    patterns = [
        r"(?:model|mod)[\s:]+([A-Za-z0-9\-]+)",
        r"\b([A-Z]{1,3}[0-9]{3,6}[A-Z]?)\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1)
    return None


def _extract_date(text: str) -> date | None:
    patterns = [
        r"(?:purchase|bought|date|dated)[\s:]+(\d{1,2}[\s/\-]?\w+[\s/\-]?\d{2,4})",
        r"(\d{1,2}[\s/\-]\w+[\s/\-]\d{2,4})",
        r"(\w+[\s/\-]\d{1,2}[\s/\-]\d{2,4})",
        r"(\d{4}[\s/\-]\d{1,2}[\s/\-]\d{1,2})",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            try:
                return date_parser.parse(match.group(1), dayfirst=True).date()
            except (ValueError, TypeError):
                continue
    return None


def _extract_amount(text: str) -> float | None:
    patterns = [
        r"(?:amount|total|price|cost|paid|rs|inr|₹)[\s.:]*([\d,]+\.?\d*)",
        r"([\d,]+\.?\d*)\s*(?:rs|inr|₹)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            amount_str = match.group(1).replace(",", "")
            try:
                return float(amount_str)
            except ValueError:
                continue
    return None


def _extract_warranty_months(text: str) -> int | None:
    match = re.search(r"(\d+)\s*(?:year|yr)s?", text, re.IGNORECASE)
    if match:
        return int(match.group(1)) * 12

    match = re.search(r"(\d+)\s*(?:month|mo)s?", text, re.IGNORECASE)
    if match:
        return int(match.group(1))

    return None


def _extract_bill_type(text: str) -> str | None:
    text_lower = text.lower()
    for bill_type in BILL_TYPES:
        if bill_type in text_lower:
            return bill_type
    return None


def _extract_provider(lines: list[str]) -> str | None:
    provider_keywords = [
        "electricity", "gas", "water", "internet", "broadband",
        "mobile", "telecom", "bsnl", "jio", "airtel", "vi",
        "adani", "tata", "reliance",
    ]
    for line in lines[:3]:
        for keyword in provider_keywords:
            if keyword.lower() in line.lower():
                return line.strip()
    return None
