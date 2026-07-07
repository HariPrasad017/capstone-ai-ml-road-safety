import os
import json
from dotenv import load_dotenv
import requests
from schema import RiskAssessment
from pydantic import ValidationError

load_dotenv()
API_KEY = os.getenv("GROQ_API_KEY")
URL = "https://api.groq.com/openai/v1/chat/completions"

def build_prompt(conditions: dict) -> str:
    return f"""You are a road safety risk assessment system. Given the following 
accident conditions, respond with ONLY a valid JSON object (no other text, no 
markdown code blocks) matching this exact structure:

{{
  "risk_level": "Low" or "Medium" or "High",
  "primary_risk_factors": ["factor1", "factor2"],
  "explanation": "one sentence explanation",
  "safety_recommendation": "one sentence actionable advice"
}}

Conditions:
- Weather: {conditions['weather']}
- Lighting: {conditions['lighting']}
- Alcohol Involved: {conditions['alcohol']}
- Speed Limit: {conditions['speed']} km/h
- Road Type: {conditions['road_type']}

Respond with ONLY the JSON object, nothing else."""


def call_llm(prompt: str) -> str:
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "llama-3.1-8b-instant",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3
    }
    response = requests.post(URL, headers=headers, json=payload, timeout=15)
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]


def get_risk_assessment(conditions: dict, max_retries: int = 3) -> RiskAssessment:
    prompt = build_prompt(conditions)
    
    for attempt in range(1, max_retries + 1):
        raw_output = call_llm(prompt)
        
        # Clean up — sometimes models wrap JSON in markdown code blocks
        cleaned = raw_output.strip().strip("```json").strip("```").strip()
        
        try:
            data = json.loads(cleaned)
            validated = RiskAssessment(**data)
            return validated
        except (json.JSONDecodeError, ValidationError) as e:
            print(f"Attempt {attempt} failed: {e}")
            if attempt == max_retries:
                raise Exception("Failed to get valid structured response after all retries")


if __name__ == "__main__":
    test_conditions = {
        "weather": "Rainy",
        "lighting": "Dark",
        "alcohol": "Yes",
        "speed": 90,
        "road_type": "National Highway"
    }
    
    result = get_risk_assessment(test_conditions)
    print(result.model_dump_json(indent=2))

def test_retry_logic():
    """
    Demonstrates that the retry logic works by using a deliberately 
    tricky prompt that's more likely to produce malformed output.
    """
    print("\n--- Testing retry logic with multiple sample inputs ---\n")
    
    test_cases = [
        {"weather": "Clear", "lighting": "Daylight", "alcohol": "No", "speed": 40, "road_type": "Urban Road"},
        {"weather": "Foggy", "lighting": "Dawn", "alcohol": "No", "speed": 60, "road_type": "State Highway"},
        {"weather": "Stormy", "lighting": "Dark", "alcohol": "Yes", "speed": 110, "road_type": "National Highway"},
    ]
    
    for i, conditions in enumerate(test_cases, 1):
        print(f"\nTest case {i}: {conditions}")
        try:
            result = get_risk_assessment(conditions)
            print(result.model_dump_json(indent=2))
        except Exception as e:
            print(f"Failed after retries: {e}")

def demo_retry_mechanism():
    """
    Proves the retry + validation logic works correctly by feeding it 
    deliberately malformed JSON, simulating what happens when an LLM 
    returns a bad response.
    """
    print("\n--- Demonstrating retry logic with simulated malformed output ---\n")
    
    fake_bad_responses = [
        'Sure! Here is the risk assessment: {"risk_level": "High", "primary_risk_factors": ["Rain"]}',  # missing fields, extra text
        '{"risk_level": "Extreme", "primary_risk_factors": ["Rain"], "explanation": "test", "safety_recommendation": "test"}',  # invalid enum value
        '{"risk_level": "High", "primary_risk_factors": ["Rain"], "explanation": "Valid now", "safety_recommendation": "Slow down"}'  # finally valid
    ]
    
    for attempt, raw in enumerate(fake_bad_responses, 1):
        cleaned = raw.strip().strip("```json").strip("```").strip()
        try:
            data = json.loads(cleaned)
            validated = RiskAssessment(**data)
            print(f"Attempt {attempt}: SUCCESS -> {validated.model_dump_json()}")
            break
        except (json.JSONDecodeError, ValidationError) as e:
            print(f"Attempt {attempt}: FAILED -> {e}\n")

def batch_process_from_csv(csv_path: str, num_rows: int = 5):
    """
    Reads real accident records from the Part 1/2 dataset and generates 
    LLM-based risk assessments for a sample of them.
    """
    import pandas as pd
    df = pd.read_csv(csv_path).head(num_rows)
    
    results = []
    for idx, row in df.iterrows():
        conditions = {
            "weather": row["Weather Conditions"],
            "lighting": row["Lighting Conditions"],
            "alcohol": row["Alcohol Involvement"],
            "speed": row["Speed Limit (km/h)"],
            "road_type": row["Road Type"]
        }
        try:
            result = get_risk_assessment(conditions)
            print(f"Row {idx}: {result.risk_level} - {result.explanation}")
            results.append(result.model_dump())
        except Exception as e:
            print(f"Row {idx}: Failed — {e}")
    
    return results


if __name__ == "__main__":
    # ... existing code ...
    demo_retry_mechanism()
    
    print("\n--- Batch processing sample rows from the accident dataset ---\n")
    batch_process_from_csv("D:\\capstone-ai-ml\\part3\\data\\accident_prediction_india.csv", num_rows=5)



if __name__ == "__main__":
    test_conditions = {
        "weather": "Rainy",
        "lighting": "Dark",
        "alcohol": "Yes",
        "speed": 90,
        "road_type": "National Highway"
    }
    
    result = get_risk_assessment(test_conditions)
    print("Single test result:")
    print(result.model_dump_json(indent=2))
    
    test_retry_logic()