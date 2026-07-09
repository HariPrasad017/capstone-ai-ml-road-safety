import pandas as pd
from llm_client import call_llm
from prompts import SYSTEM_PROMPT, build_user_prompt

df = pd.read_csv('data/cleaned_data.csv')
sample_records = df[['Weather Conditions', 'Lighting Conditions', 'Alcohol Involvement', 
                      'Speed Limit (km/h)', 'Road Type']].head(3).to_dict('records')

comparison_results = []

for i, record in enumerate(sample_records, 1):
    user_prompt = build_user_prompt(record)
    
    output_temp0 = call_llm(SYSTEM_PROMPT, user_prompt, temperature=0.0)
    output_temp07 = call_llm(SYSTEM_PROMPT, user_prompt, temperature=0.7)
    
    print(f"\n--- Record {i} ---")
    print(f"Input: {record}")
    print(f"Temp=0.0: {output_temp0}")
    print(f"Temp=0.7: {output_temp07}")
    
    comparison_results.append({
        'Input': record,
        'Output (temp=0)': output_temp0,
        'Output (temp=0.7)': output_temp07
    })