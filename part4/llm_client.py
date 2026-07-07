import os
import json
import requests
from dotenv import load_dotenv
from pydantic import ValidationError as PydanticValidationError
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from schema import RiskAssessment
from logger_setup import setup_logger
import time


_last_request_time = 0
_min_seconds_between_requests = 2  
_total_requests = 0
_total_tokens_used = 0



load_dotenv()
API_KEY = os.getenv("GROQ_API_KEY")
URL = "https://api.groq.com/openai/v1/chat/completions"
logger = setup_logger()


class LLMCallError(Exception):
    pass


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


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type((LLMCallError, json.JSONDecodeError, PydanticValidationError)),
    reraise=True
)
def get_risk_assessment(conditions: dict) -> RiskAssessment:
    global _last_request_time, _total_requests, _total_tokens_used
    
    elapsed = time.time() - _last_request_time
    if elapsed < _min_seconds_between_requests:
        wait_time = _min_seconds_between_requests - elapsed
        logger.info(f"Rate limit: waiting {wait_time:.1f}s before next request")
        time.sleep(wait_time)
    _last_request_time = time.time()
    
    prompt = build_prompt(conditions)
    logger.info(f"Sending request to LLM with conditions: {conditions}")
    
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "llama-3.1-8b-instant",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3
    }
    
    try:
        response = requests.post(URL, headers=headers, json=payload, timeout=15)
        response.raise_for_status()
    except requests.exceptions.Timeout:
        logger.error("Request timed out")
        raise LLMCallError("LLM request timed out after 15 seconds")
    except requests.exceptions.RequestException as e:
        logger.error(f"Request failed: {e}")
        raise LLMCallError(f"API request failed: {e}")
    
    response_json = response.json()
    raw_output = response_json["choices"][0]["message"]["content"]
    
    
    _total_requests += 1
    usage = response_json.get("usage", {})
    tokens_this_call = usage.get("total_tokens", 0)
    _total_tokens_used += tokens_this_call
    logger.info(f"Usage: {tokens_this_call} tokens this call | Running total: {_total_requests} requests, {_total_tokens_used} tokens")
    
    cleaned = raw_output.strip().strip("```json").strip("```").strip()
    
    try:
        data = json.loads(cleaned)
        validated = RiskAssessment(**data)
        logger.info(f"Successfully received valid response: {validated.risk_level}")
        return validated
    except (json.JSONDecodeError, PydanticValidationError) as e:
        logger.warning(f"Invalid response format, will retry: {e}")
        raise


def get_usage_stats():
    """Returns cumulative usage stats for this session."""
    return {"total_requests": _total_requests, "total_tokens": _total_tokens_used}