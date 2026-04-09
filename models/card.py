from pydantic import BaseModel, Field
from typing import List

class CaseStudyCard(BaseModel):
    """
    Structured representation of a Project Management Case Study Card.
    """
    concept: str = Field(..., description="The PM concept being illustrated")
    story: str = Field(..., description="A 2-3 sentence real-world scenario")
    problem: str = Field(..., description="What went wrong or what decision was needed")
    decision_point: str = Field(..., description="The critical PM moment in the scenario")
    concept_mapping: str = Field(..., description="How the scenario maps back to the PM concept")
    key_lessons: List[str] = Field(..., description="3 bullet points the student should take away")
    think_about_this: str = Field(..., description="One follow-up question to deepen understanding")
