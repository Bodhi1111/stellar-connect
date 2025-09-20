# src/data_models.py
from pydantic import BaseModel, Field
from typing import Optional, Literal, List

# Section 4.1
class SalesRecord(BaseModel):
    """A structured record of a sales meeting."""
    client_name: str = Field(description="The full name of the client or company.")
    meeting_date: Optional[str] = Field(description="The date of the meeting in YYYY-MM-DD format.", default=None)
    estate_value: Optional[float] = Field(description="The estimated estate value or deal size, if mentioned.", default=None)
    outcome: Literal["closed won", "follow up", "negotiation", "closed lost", "undetermined"] = Field(
        description="The final outcome or current status of the meeting."
    )
    summary: str = Field(description="A concise, one-paragraph summary of the meeting's key discussion points.")
    action_items: List[str] = Field(description="A list of specific action items or next steps.")

# Section 4.3
class TestimonialQuote(BaseModel):
    """A compelling quote suitable for marketing purposes."""
    quote: str = Field(description="The verbatim quote from the client.")
    context: str = Field(description="A brief explanation of the context.")
    potential_use_case: str = Field(description="Suggested marketing use case (e.g., 'Website', 'Social Media').")

class ViralContent(BaseModel):
    quotes: List[TestimonialQuote]