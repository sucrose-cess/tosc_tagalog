
import pandas as pd
import requests

df = pd.read_csv(r'C:\Users\rualryt\OneDrive\Documents\taglish_tosc\data\gold_dataset.csv')
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "gemma2:2b"
results = []

for index, row in df.iterrows():
    text = row['text']
    if pd.isna(text):
        continue
    text = str(text)
    prompt = f"""Classify the sentiment of this Taglish political text toward the mentioned entity as Positive, Negative, or Neutral. Reply with one word only: Positive, Negative, or Neutral.

Text: {text}
Sentiment:"""
    response = requests.post(OLLAMA_URL, json={"model": MODEL, "prompt": prompt, "stream": False})
    prediction = response.json().get('response', 'Error').strip()
    results.append({'text': text, 'true_label': row['final_label'], 'predicted': prediction})
    print(f"Row {index}: {prediction}")

print(pd.DataFrame(results).to_string())
print("Done!")
