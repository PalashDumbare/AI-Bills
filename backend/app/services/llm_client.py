import json
import httpx
from ..config import get_settings


EXTRACTION_PROMPT = """Extract structured data from this bill/invoice text.

Return ONLY a JSON object with these fields (use null for fields not found):

For appliance invoices:
{{
  "document_type": "appliance_invoice",
  "brand": "string or null",
  "product": "string or null",
  "model": "string or null",
  "purchase_date": "YYYY-MM-DD or null",
  "amount": number or null,
  "warranty_months": number or null
}}

For utility bills:
{{
  "document_type": "bill",
  "provider": "string or null",
  "bill_type": "electricity|gas|water|internet|mobile|broadband or null",
  "amount": number or null
}}

Text to analyze:
{text}

Return ONLY the JSON object, no explanation:"""


async def extract_with_llm(text: str) -> dict | None:
    """Extract structured data using Ollama LLM."""
    try:
        settings = get_settings()

        prompt = EXTRACTION_PROMPT.format(text=text)

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{settings.ollama_base_url}/api/generate",
                json={
                    "model": settings.ollama_model,
                    "prompt": prompt,
                    "stream": False,
                    "format": "json",
                },
            )
            response.raise_for_status()

            result = response.json()
            content = result.get("response", "")

            # Parse JSON from LLM response
            try:
                # Try to extract JSON from the response
                content = content.strip()
                if content.startswith("```"):
                    content = content.split("```")[1]
                    if content.startswith("json"):
                        content = content[4:]
                data = json.loads(content)
                return _normalize_response(data)
            except (json.JSONDecodeError, IndexError):
                return None
    except Exception:
        return None


def _normalize_response(data: dict) -> dict | None:
    """Normalize LLM response to match expected format."""
    from datetime import date
    from dateutil import parser as date_parser

    doc_type = data.get("document_type")
    if doc_type not in ("appliance_invoice", "bill"):
        return None

    result = {"document_type": doc_type}

    if doc_type == "appliance_invoice":
        result["brand"] = data.get("brand")
        result["product"] = data.get("product")
        result["model"] = data.get("model")

        # Parse purchase date
        purchase_date = data.get("purchase_date")
        if purchase_date:
            try:
                result["purchase_date"] = date_parser.parse(purchase_date).date()
            except (ValueError, TypeError):
                result["purchase_date"] = None
        else:
            result["purchase_date"] = None

        # Parse amount
        amount = data.get("amount")
        if amount is not None:
            try:
                result["amount"] = float(amount)
            except (ValueError, TypeError):
                result["amount"] = None
        else:
            result["amount"] = None

        # Parse warranty months
        warranty = data.get("warranty_months")
        if warranty is not None:
            try:
                result["warranty_months"] = int(warranty)
            except (ValueError, TypeError):
                result["warranty_months"] = None
        else:
            result["warranty_months"] = None

        # Calculate warranty expiry
        if result.get("warranty_months") and result.get("purchase_date"):
            from dateutil.relativedelta import relativedelta
            result["warranty_expiry"] = (
                result["purchase_date"] + relativedelta(months=result["warranty_months"])
            )
        else:
            result["warranty_expiry"] = None

    else:  # bill
        result["provider"] = data.get("provider")
        result["bill_type"] = data.get("bill_type")

        amount = data.get("amount")
        if amount is not None:
            try:
                result["amount"] = float(amount)
            except (ValueError, TypeError):
                result["amount"] = None
        else:
            result["amount"] = None

    return result
