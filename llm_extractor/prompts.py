"""Parameterised prompt templates with few-shot examples.

Templates use `string.Template` ($placeholders) so the JSON braces in
examples never clash with Python's str.format.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from string import Template

from pydantic import BaseModel

SYSTEM_TEMPLATE = Template(
    """You are a precise $doc_type extraction engine.
Return ONLY a single JSON object. No markdown, no code fences, no commentary.
The JSON must validate against this JSON Schema:
$schema
Rules:
- Use null for fields that are not present. Never invent values.
- Dates must be ISO 8601 (YYYY-MM-DD) unless the schema says otherwise.
- Numbers must be plain numbers (no currency symbols or thousands separators).
$extra_rules"""
)

USER_TEMPLATE = Template(
    """$examples

Now extract from this $doc_type:
<document>
$document
</document>
JSON:"""
)

RETRY_TEMPLATE = Template(
    """Your previous answer failed validation:
$errors
Previous answer:
$previous
Return a corrected JSON object only."""
)


@dataclass
class FewShotExample:
    document: str
    output: dict


@dataclass
class PromptTemplate:
    doc_type: str
    schema: type[BaseModel]
    examples: list[FewShotExample] = field(default_factory=list)
    extra_rules: list[str] = field(default_factory=list)

    def system(self) -> str:
        return SYSTEM_TEMPLATE.substitute(
            doc_type=self.doc_type,
            schema=json.dumps(self.schema.model_json_schema(), indent=2),
            extra_rules="\n".join(f"- {r}" for r in self.extra_rules),
        )

    def user(self, document: str) -> str:
        shots = "\n\n".join(
            f"Example {i}:\n<document>\n{ex.document}\n</document>\nJSON:\n{json.dumps(ex.output)}"
            for i, ex in enumerate(self.examples, 1)
        )
        return USER_TEMPLATE.substitute(examples=shots, doc_type=self.doc_type, document=document.strip())

    @staticmethod
    def retry(errors: str, previous: str) -> str:
        return RETRY_TEMPLATE.substitute(errors=errors, previous=previous)


# ------------------------------------------------------------ ready-made templates
INVOICE_EXAMPLE = FewShotExample(
    document="""ACME Supplies Ltd - Invoice #INV-1001
Date: 03/02/2026   Due: 05/03/2026
Bill to: Northwind Cafe
2 x Coffee grinder @ £45.00 = £90.00
1 x Descaler pack @ £10.00 = £10.00
Subtotal £100.00  VAT £20.00  Total £120.00""",
    output={
        "invoice_number": "INV-1001",
        "vendor_name": "ACME Supplies Ltd",
        "customer_name": "Northwind Cafe",
        "issue_date": "2026-02-03",
        "due_date": "2026-03-05",
        "currency": "GBP",
        "line_items": [
            {"description": "Coffee grinder", "quantity": 2, "unit_price": 45.00, "total": 90.00},
            {"description": "Descaler pack", "quantity": 1, "unit_price": 10.00, "total": 10.00},
        ],
        "subtotal": 100.00,
        "tax": 20.00,
        "total": 120.00,
    },
)

RESUME_EXAMPLE = FewShotExample(
    document="""Jane Doe | jane.doe@example.com | 07700 900123
Data analyst with SQL and Python experience.
Skills: Python, SQL, Tableau, python
Experience: Junior Data Analyst, Fabrikam - Jun 2023 to present. Built weekly sales dashboards.
Education: BSc Mathematics, University of Leeds, 2023""",
    output={
        "full_name": "Jane Doe",
        "email": "jane.doe@example.com",
        "phone": "07700 900123",
        "summary": "Data analyst with SQL and Python experience.",
        "skills": ["Python", "SQL", "Tableau"],
        "experience": [
            {
                "job_title": "Junior Data Analyst",
                "company": "Fabrikam",
                "start_date": "2023-06",
                "end_date": "present",
                "highlights": ["Built weekly sales dashboards."],
            }
        ],
        "education": [
            {"institution": "University of Leeds", "qualification": "BSc Mathematics", "graduation_year": 2023}
        ],
        "years_of_experience": 3.3,
    },
)


def build_template(doc_type: str, today: str) -> PromptTemplate:
    from .schemas import Invoice, Resume

    if doc_type == "invoice":
        return PromptTemplate(
            "invoice", Invoice, [INVOICE_EXAMPLE],
            ["UK dates written DD/MM/YYYY must be converted to YYYY-MM-DD.",
             "Infer currency from symbols: £=GBP, $=USD, €=EUR."],
        )
    if doc_type == "resume":
        return PromptTemplate(
            "resume", Resume, [RESUME_EXAMPLE],
            [f"Today's date is {today}; use it to compute years_of_experience for 'present' roles.",
             "Deduplicate skills case-insensitively."],
        )
    raise ValueError(f"Unknown doc_type {doc_type!r}; choose 'invoice' or 'resume'.")
