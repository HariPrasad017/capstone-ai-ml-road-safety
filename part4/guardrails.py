import re

def has_pii(text):
    email_pattern = r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+'
    phone_pattern = r'\b\d{10}\b|\b\d{3}[-.\s]\d{3}[-.\s]\d{4}\b'
    return bool(re.search(email_pattern, text) or re.search(phone_pattern, text))


if __name__ == "__main__":
    test_with_email = "Contact driver at john.doe@email.com for details"
    test_clean = "Weather was rainy, speed limit 80 km/h"

    print(f"Test with email: '{test_with_email}'")
    print(f"PII detected: {has_pii(test_with_email)}")

    print(f"\nTest clean input: '{test_clean}'")
    print(f"PII detected: {has_pii(test_clean)}")