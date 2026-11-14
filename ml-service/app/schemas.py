"""Strict public API contract for Avito hackathon."""
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, StringConstraints
from typing import Annotated

CategoryId = Literal[101, 102, 103, 104, 105, 106]

class GenerateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    description: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=12000)]
    mcId: CategoryId

class Draft(BaseModel):
    model_config = ConfigDict(extra="forbid")
    mcId: CategoryId
    title: str | None = Field(default=None, max_length=50)
    text: str | None = None

class GenerateResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    drafts: list[Draft]
