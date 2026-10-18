"""Offline tests using MockProvider - no API key needed. Run: pytest -q"""
import json

import pytest

from llm_extractor import ExtractionError, Invoice, MockProvider, Resume, analyse

GOOD_INVOICE = {
    "invoice_number": "BB-2026-0457",
    "vendor_name": "BrightByte IT Services",
    "customer_name": "Coventry Student Union",
    "issue_date": "2026-10-14",
    "due_date": "2026-11-13",
    "currency": "GBP",
    "line_items": [
        {"description": "Laptop repair (labour hrs)", "quantity": 3, "unit_price": 35, "total": 105},
        {"description": "SSD 1TB", "quantity": 2, "unit_price": 62.5, "total": 125},
        {"description": "Data backup service", "quantity": 1, "unit_price": 20, "total": 20},
    ],
    "subtotal": 250, "tax": 50, "total": 300,
}


def test_valid_invoice_first_try():
    provider = MockProvider([GOOD_INVOICE])
    inv = analyse("doc", "invoice", provider)
    assert isinstance(inv, Invoice) and str(inv.total) == "300"
    assert len(provider.calls) == 1


def test_code_fences_are_stripped():
    provider = MockProvider(["```json\n" + json.dumps(GOOD_INVOICE) + "\n```"])
    assert analyse("doc", "invoice", provider).invoice_number == "BB-2026-0457"


def test_bad_total_triggers_retry_with_errors():
    bad = {**GOOD_INVOICE, "total": 999}
    provider = MockProvider([bad, GOOD_INVOICE])
    inv = analyse("doc", "invoice", provider)
    assert str(inv.total) == "300"
    retry_msg = provider.calls[1][-1]["content"]
    assert "failed validation" in retry_msg and "subtotal + tax" in retry_msg


def test_extra_keys_rejected():
    provider = MockProvider([{**GOOD_INVOICE, "notes": "made up"}] * 3)
    with pytest.raises(ExtractionError) as e:
        analyse("doc", "invoice", provider, max_retries=2)
    assert len(e.value.attempts) == 3


def test_invalid_json_then_recovers():
    provider = MockProvider(["Sorry, here it is: {not json", GOOD_INVOICE])
    assert analyse("doc", "invoice", provider).currency == "GBP"


def test_resume_validation_and_skill_dedupe():
    provider = MockProvider([{
        "full_name": "Sam Patel",
        "email": "sam.patel@example.com",
        "phone": "+44 7700 900456",
        "summary": None,
        "skills": ["Python", "Pandas", "pandas", "SQL"],
        "experience": [{"job_title": "Python Programming Intern", "company": "Example Tech Ltd",
                        "start_date": "2026-07", "end_date": "2026-09", "highlights": []}],
        "education": [{"institution": "Coventry University London",
                       "qualification": "BSc Computing Science", "graduation_year": 2028}],
        "years_of_experience": 2.8,
    }])
    r = analyse("doc", "resume", provider)
    assert isinstance(r, Resume) and r.skills == ["Python", "Pandas", "SQL"]


def test_bad_email_rejected():
    bad = {"full_name": "X", "email": "not-an-email", "years_of_experience": 1}
    with pytest.raises(ExtractionError):
        analyse("doc", "resume", MockProvider([bad] * 3))
