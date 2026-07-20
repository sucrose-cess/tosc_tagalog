import pandas as pd
import requests
import json

df = pd.read_csv(r'C:\Users\rualryt\OneDrive\Documents\taglish_tosc\data\stress_dataset.csv')

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "llama3.2:3b"

results = []

for index, row in df.iterrows():
    text = row['text']
    
    prompt = f"""You are a sentiment classifier. Given the following Taglish (Tagalog-English) political text, classify the sentiment as Positive, Negative, or Neutral. Reply with one word only.

Text: {text}
Sentiment:"""

    response = requests.post(OLLAMA_URL, json={
        "model": MODEL,
        "prompt": prompt,
        "stream": False
    })
    
    prediction = response.json()['response'].strip()
    
    results.append({
        'text': text,
        'true_label': row['final_label'],
        'predicted': prediction
    })
    
    print(f"Row {index}: {prediction}")

output = pd.DataFrame(results)
output.to_csv(r'C:\Users\rualryt\OneDrive\Documents\taglish_tosc\results\llama_without_calamancy_stress.csv', index=False)
print("Done!")