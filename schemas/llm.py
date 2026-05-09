from pydantic import BaseModel, Field, StrictInt, StrictStr


class ChatRequest(BaseModel):
    message: str = Field(..., description="User message")
    system_prompt: str | None = Field(default=None, description="Optional system prompt")


class ChatResponse(BaseModel):
    answer: str


class SummaryOutput(BaseModel):
    title: StrictStr = Field(..., description="Summary title")
    summary: StrictStr = Field(..., description="Summary text")
    key_points: list[StrictStr] = Field(default_factory=list, description="Key points")
    action_items: list[StrictStr] = Field(default_factory=list, description="Action items")


class PersonInfoOutput(BaseModel):
    name: StrictStr = Field(..., description="Person name")
    age: StrictInt = Field(..., description="Person age")
    skills: list[StrictStr] = Field(default_factory=list, description="Skills")


