from pydantic import BaseModel
from typing import List
from enum import Enum

class RiskLevel(str, Enum):
    low = "Low"
    medium = "Medium"
    high = "High"

class RiskAssessment(BaseModel):
    risk_level: RiskLevel
    primary_risk_factors: List[str]
    explanation: str
    safety_recommendation: str