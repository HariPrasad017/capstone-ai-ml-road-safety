from validators import validate_conditions, ValidationError

def run_test(description, input_data, should_pass):
    try:
        result = validate_conditions(input_data)
        status = "PASS" if should_pass else "FAIL (expected error, got success)"
    except ValidationError as e:
        status = "PASS" if not should_pass else f"FAIL (expected success, got error: {e})"
    print(f"[{status}] {description}")


if __name__ == "__main__":
    print("Running validation tests...\n")
    
    # Valid input — should pass
    run_test(
        "Valid complete input",
        {"weather": "Rainy", "lighting": "Dark", "alcohol": "Yes", "speed": "90", "road_type": "Highway"},
        should_pass=True
    )
    
    # Invalid weather — should fail
    run_test(
        "Invalid weather value",
        {"weather": "Sunny", "lighting": "Dark", "alcohol": "Yes", "speed": "90", "road_type": "Highway"},
        should_pass=False
    )
    
    # Speed as text — should fail
    run_test(
        "Non-numeric speed",
        {"weather": "Clear", "lighting": "Dark", "alcohol": "Yes", "speed": "abc", "road_type": "Highway"},
        should_pass=False
    )
    
    # Speed out of range — should fail
    run_test(
        "Speed out of valid range (300)",
        {"weather": "Clear", "lighting": "Dark", "alcohol": "Yes", "speed": "300", "road_type": "Highway"},
        should_pass=False
    )
    
    # Empty road type — should fail
    run_test(
        "Empty road type",
        {"weather": "Clear", "lighting": "Dark", "alcohol": "Yes", "speed": "60", "road_type": ""},
        should_pass=False
    )
    
    # Lowercase input — should pass (title() handles it)
    run_test(
        "Lowercase input normalized correctly",
        {"weather": "clear", "lighting": "dark", "alcohol": "yes", "speed": "60", "road_type": "Urban Road"},
        should_pass=True
    )
    
    print("\nAll tests completed.")