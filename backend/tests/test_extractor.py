"""Unit tests for the rule-based extractor."""
from datetime import date
from app.services.extractor import (
    extract_structured_data,
    _extract_brand,
    _extract_product,
    _extract_model,
    _extract_date,
    _extract_amount,
    _extract_warranty_months,
    _extract_bill_type,
    _extract_provider,
)

SAMPLE_APPLIANCE_TEXT = """LG Washing Machine
Purchase Date: 15 Jan 2026
Amount: 32999
Model: FHM1207
Warranty: 2 years"""

SAMPLE_BILL_TEXT = """Electricity Bill
Provider: BSES Rajdhani
Amount: 1500
Due Date: 20 Feb 2026"""


class TestBrandExtraction:
    def test_known_brand(self):
        lines = ["LG Washing Machine", "Some other line"]
        assert _extract_brand(lines) == "LG"

    def test_brand_case_insensitive(self):
        lines = ["samsung tv", "Some other line"]
        assert _extract_brand(lines) == "Samsung"

    def test_no_brand(self):
        lines = ["Some random text", "Another line"]
        assert _extract_brand(lines) is None

    def test_brand_in_first_three_lines(self):
        lines = ["Line 1", "Line 2", "Sony Camera", "Line 4"]
        assert _extract_brand(lines) == "Sony"


class TestProductExtraction:
    def test_washing_machine(self):
        lines = ["LG Washing Machine", "Some other line"]
        assert _extract_product(lines) == "Washing Machine"

    def test_refrigerator(self):
        lines = ["Samsung Refrigerator", "Some other line"]
        assert _extract_product(lines) == "Refrigerator"

    def test_no_product(self):
        lines = ["Some random text", "Another line"]
        assert _extract_product(lines) is None


class TestModelExtraction:
    def test_model_keyword(self):
        text = "Model: FHM1207"
        assert _extract_model(text) == "FHM1207"

    def test_model_keyword_lowercase(self):
        text = "model: ABC123"
        assert _extract_model(text) == "ABC123"

    def test_alphanumeric_code(self):
        text = "Some text X12345 more text"
        assert _extract_model(text) == "X12345"

    def test_no_model(self):
        text = "Some random text without model"
        assert _extract_model(text) is None


class TestDateExtraction:
    def test_date_with_keyword(self):
        text = "Purchase Date: 15 Jan 2026"
        result = _extract_date(text)
        assert result == date(2026, 1, 15)

    def test_date_formats(self):
        text = "Date: 15/01/2026"
        result = _extract_date(text)
        assert result == date(2026, 1, 15)

    def test_no_date(self):
        text = "Some text without date"
        assert _extract_date(text) is None


class TestAmountExtraction:
    def test_amount_with_keyword(self):
        text = "Amount: 32999"
        assert _extract_amount(text) == 32999.0

    def test_amount_with_currency(self):
        text = "Price: Rs. 1500"
        assert _extract_amount(text) == 1500.0

    def test_amount_with_commas(self):
        text = "Total: 1,50,000"
        assert _extract_amount(text) == 150000.0

    def test_no_amount(self):
        text = "Some text without amount"
        assert _extract_amount(text) is None


class TestWarrantyExtraction:
    def test_years(self):
        text = "Warranty: 2 years"
        assert _extract_warranty_months(text) == 24

    def test_months(self):
        text = "Warranty: 6 months"
        assert _extract_warranty_months(text) == 6

    def test_no_warranty(self):
        text = "Some text without warranty"
        assert _extract_warranty_months(text) is None


class TestBillTypeExtraction:
    def test_electricity(self):
        text = "Electricity bill for January"
        assert _extract_bill_type(text) == "electricity"

    def test_gas(self):
        text = "Gas bill payment"
        assert _extract_bill_type(text) == "gas"

    def test_no_bill_type(self):
        text = "Some random text"
        assert _extract_bill_type(text) is None


class TestProviderExtraction:
    def test_provider(self):
        lines = ["BSES Rajdhani Electricity", "Some other line"]
        assert _extract_provider(lines) == "BSES Rajdhani Electricity"

    def test_no_provider(self):
        lines = ["Some random text", "Another line"]
        assert _extract_provider(lines) is None


class TestFullExtraction:
    def test_appliance_invoice(self):
        result = extract_structured_data(SAMPLE_APPLIANCE_TEXT)
        assert result is not None
        assert result["document_type"] == "appliance_invoice"
        assert result["brand"] == "LG"
        assert result["product"] == "Washing Machine"
        assert result["model"] == "FHM1207"
        assert result["purchase_date"] == date(2026, 1, 15)
        assert result["amount"] == 32999.0
        assert result["warranty_months"] == 24
        assert result["warranty_expiry"] == date(2028, 1, 15)

    def test_bill(self):
        result = extract_structured_data(SAMPLE_BILL_TEXT)
        assert result is not None
        assert result["document_type"] == "bill"
        assert result["bill_type"] == "electricity"
        assert result["amount"] == 1500.0

    def test_empty_text(self):
        result = extract_structured_data("")
        assert result is None

    def test_unrecognized_text(self):
        result = extract_structured_data("Just some random text with no structure")
        assert result is None
