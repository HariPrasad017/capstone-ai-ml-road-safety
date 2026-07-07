VALID_WEATHER = {"Clear", "Rainy", "Foggy", "Stormy", "Hazy"}
VALID_LIGHTING = {"Daylight", "Dark", "Dawn", "Dusk"}
VALID_ALCOHOL = {"Yes", "No"}

class ValidationError(Exception):
    pass

def validate_conditions(conditions: dict) -> dict:
    """
    Validates and sanitizes user input before sending to the LLM.
    Raises ValidationError with a clear message if anything is invalid.
    """
    errors = []
    
    weather = conditions.get('weather', '').strip().title()
    if weather not in VALID_WEATHER:
        errors.append(f"Weather must be one of {VALID_WEATHER}, got '{weather}'")
    
    lighting = conditions.get('lighting', '').strip().title()
    if lighting not in VALID_LIGHTING:
        errors.append(f"Lighting must be one of {VALID_LIGHTING}, got '{lighting}'")
    
    alcohol = conditions.get('alcohol', '').strip().title()
    if alcohol not in VALID_ALCOHOL:
        errors.append(f"Alcohol must be one of {VALID_ALCOHOL}, got '{alcohol}'")
    
    try:
        speed = int(conditions.get('speed', 0))
        if not (0 < speed <= 200):
            errors.append(f"Speed must be between 1-200 km/h, got {speed}")
    except (ValueError, TypeError):
        errors.append(f"Speed must be a number, got '{conditions.get('speed')}'")
        speed = 0
    
    road_type = conditions.get('road_type', '').strip()
    if not road_type:
        errors.append("Road type cannot be empty")
    
    if errors:
        raise ValidationError("; ".join(errors))
    
    return {
        "weather": weather,
        "lighting": lighting,
        "alcohol": alcohol,
        "speed": speed,
        "road_type": road_type
    }