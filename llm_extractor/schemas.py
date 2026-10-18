"""Pydantic schemas that every LLM response must satisfy.

`extra="forbid"` rejects any keys the model invents, and the validators
check that the numbers are internally consistent rather than just well typed.
"""
from __future__ import annotations

from datetime import date
from decimal import Decimal
from typing import Optional

from pydantic import BaseModel, ConfigDict, EmailStr, Field, field_validator, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


# ---------------------------------------------------------------- Invoice
class LineItem(StrictModel):
    description: str = Field(min_length=1)
    quantity: Decimal = Field(gt=0)
    unit_price: Decimal = Field(ge=0)
    total: Decimal = Field(ge=0)

    @model_validator(mode="after")
    def check_line_total(self) -> "LineItem":
        expected = (self.quantity * self.unit_price).quantize(Decimal("0.01"))
        if abs(expected - self.total) > Decimal("0.01"):
            raise ValueError(
                f"line total {self.total} != quantity x unit_price ({expected})"
            )
        return self


class Invoice(StrictModel):
    invoice_number: str = Field(min_length=1)
    vendor_name: str = Field(min_length=1)
    customer_name: Optional[str] = None
    issue_date: date
    due_date: Optional[date] = None
    currency: str = Field(pattern=r"^[A-Z]{3}$", description="ISO 4217 code, e.g. GBP")
    line_items: list[LineItem] = Field(min_length=1)
    subtotal: Decimal = Field(ge=0)
    tax: Decimal = Field(ge=0)
    total: Decimal = Field(ge=0)

    @model_validator(mode="after")
    def check_totals(self) -> "Invoice":
        items_sum = sum((i.total for i in self.line_items), Decimal("0"))
        if abs(items_sum - self.subtotal) > Decimal("0.01"):
            raise ValueError(f"subtotal {self.subtotal} != sum of line items ({items_sum})")
        if abs(self.subtotal + self.tax - self.total) > Decimal("0.01"):
            raise ValueError(f"total {self.total} != subtotal + tax ({self.subtotal + self.tax})")
        if self.due_date and self.due_date < self.issue_date:
            raise ValueError("due_date is before issue_date")
        return self


# ---------------------------------------------------------------- Resume
class Experience(StrictModel):
    job_title: str
    company: str
    start_date: str = Field(pattern=r"^\d{4}(-\d{2})?$", description="YYYY or YYYY-MM")
    end_date: Optional[str] = Field(
        default=None, pattern=r"^(\d{4}(-\d{2})?|present)$", description="YYYY, YYYY-MM or 'present'"
    )
    highlights: list[str] = Field(default_factory=list)


class Education(StrictModel):
    institution: str
    qualification: str
    graduation_year: Optional[int] = Field(default=None, ge=1950, le=2100)


class Resume(StrictModel):
    full_name: str = Field(min_length=1)
    email: Optional[EmailStr] = None
    phone: Optional[str] = None
    summary: Optional[str] = None
    skills: list[str] = Field(default_factory=list)
    experience: list[Experience] = Field(default_factory=list)
    education: list[Education] = Field(default_factory=list)
    years_of_experience: float = Field(ge=0, le=60)

    @field_validator("skills")
    @classmethod
    def dedupe_skills(cls, v: list[str]) -> list[str]:
        seen, out = set(), []
        for s in v:
            key = s.strip().lower()
            if key and key not in seen:
                seen.add(key)
                out.append(s.strip())
        return out


SCHEMAS: dict[str, type[StrictModel]] = {"invoice": Invoice, "resume": Resume}
