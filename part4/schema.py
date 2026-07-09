RISK_ASSESSMENT_SCHEMA = {
    "type": "object",
    "properties": {
        "risk_tier": {"type": "string", "enum": ["low", "medium", "high"]},
        "flag_for_review": {"type": "boolean"},
        "primary_signal": {"type": "string"},
        "confidence": {"type": "string", "enum": ["low", "medium", "high"]},
        "recommended_action": {"type": "string"}
    },
    "required": ["risk_tier", "flag_for_review", "primary_signal", "confidence", "recommended_action"]
}