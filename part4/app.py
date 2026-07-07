from validators import validate_conditions, ValidationError
from llm_client import get_risk_assessment, LLMCallError
from logger_setup import setup_logger
from llm_client import get_risk_assessment, LLMCallError, get_usage_stats

logger = setup_logger()


def print_banner():
    print("=" * 55)
    print("   ROAD SAFETY ADVISOR — AI Risk Assessment System")
    print("=" * 55)
    print("Enter accident conditions to get an AI risk assessment.")
    print("Type 'quit' at any prompt to exit.\n")


def get_user_input():
    """Collects raw input from the user for one assessment."""
    weather = input("Weather (Clear/Rainy/Foggy/Stormy/Hazy): ")
    if weather.lower() == 'quit':
        return None
    
    lighting = input("Lighting (Daylight/Dark/Dawn/Dusk): ")
    alcohol = input("Alcohol Involved (Yes/No): ")
    speed = input("Speed Limit (km/h): ")
    road_type = input("Road Type (e.g. National Highway): ")
    
    return {
        "weather": weather,
        "lighting": lighting,
        "alcohol": alcohol,
        "speed": speed,
        "road_type": road_type
    }


def display_result(result):
    print("\n--- Risk Assessment ---")
    print(f"Risk Level        : {result.risk_level.value}")
    print(f"Contributing Factors: {', '.join(result.primary_risk_factors)}")
    print(f"Explanation       : {result.explanation}")
    print(f"Recommendation    : {result.safety_recommendation}")
    print("-" * 40 + "\n")


def main():
    print_banner()
    
    while True:
        try:
            raw_input_data = get_user_input()
            if raw_input_data is None:
               stats = get_usage_stats()
               print(f"\nSession summary: {stats['total_requests']} requests, {stats['total_tokens']} tokens used")
               print("Goodbye! Stay safe on the road.")
               logger.info(f"User exited. Session stats: {stats}")
               break
            
            # Step 1: Validate input
            try:
                clean_conditions = validate_conditions(raw_input_data)
            except ValidationError as e:
                print(f"\n  Invalid input: {e}\n")
                logger.warning(f"Input validation failed: {e}")
                continue  # ask again, don't crash
            
            # Step 2: Call LLM (with retries/timeout handled inside)
            print("\nAnalyzing risk... please wait.\n")
            try:
                result = get_risk_assessment(clean_conditions)
                display_result(result)
            except LLMCallError as e:
                print(f"\nCould not get a risk assessment right now: {e}")
                print("Please try again in a moment.\n")
                logger.error(f"LLM call failed after retries: {e}")
            except Exception as e:
                print(f"\n Unexpected error: {e}\n")
                logger.error(f"Unexpected error: {e}")
        
        except KeyboardInterrupt:
            print("\n\nInterrupted. Goodbye!")
            logger.info("User interrupted with Ctrl+C")
            break


if __name__ == "__main__":
    main()