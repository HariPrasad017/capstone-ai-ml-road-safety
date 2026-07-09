import json
import pandas as pd
from jsonschema import validate, ValidationError
from llm_client import call_llm
from guardrails import has_pii
from prompts import SYSTEM_PROMPT, build_user_prompt
from schema import RISK_ASSESSMENT_SCHEMA

FALLBACK = {
    "risk_tier": None,
    "flag_for_review": None,
    "primary_signal": None,
    "confidence": None,
    "recommended_action": None
}

def score_record(record):
    user_prompt = build_user_prompt(record)
    
    if has_pii(user_prompt):
        print("Input blocked: PII detected.")
        return None
    
    raw_output = call_llm(SYSTEM_PROMPT, user_prompt, temperature=0.0)
    
    try:
        cleaned = raw_output.strip().strip("```json").strip("```").strip()
        parsed = json.loads(cleaned)
    except json.JSONDecodeError as e:
        print(f"JSON parse failed: {e}")
        return FALLBACK
    
    try:
        validate(instance=parsed, schema=RISK_ASSESSMENT_SCHEMA)
        return parsed
    except ValidationError as e:
        print(f"Schema validation failed: {e.message}")
        return FALLBACK


if __name__ == "__main__":
    df = pd.read_csv('data/cleaned_data.csv')
    
    sample_records = df[['Weather Conditions', 'Lighting Conditions', 'Alcohol Involvement', 
                          'Speed Limit (km/h)', 'Road Type']].head(3).to_dict('records')
    
    results_table = []
    
    for i, record in enumerate(sample_records, 1):
        print(f"\n--- Record {i} ---")
        print(f"Input: {record}")
        
        result = score_record(record)
        status = 'PASS' if result and result.get('risk_tier') is not None else 'FAIL'
        
        print(f"Validation Outcome: {status}")
        print(f"Assessment: {result}")
        
        results_table.append({
            'Input Record': record,
            'LLM Assessment JSON': result,
            'Validation Status': status
        })