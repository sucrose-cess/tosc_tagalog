import pandas as pd
import requests
import calamancy

df = pd.read_csv(r'C:\Users\rualryt\OneDrive\Documents\taglish_tosc\data\stress_dataset_with_entities.csv')

nlp = calamancy.load('tl_calamancy_md-0.2.0')

# Ollama settings
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "llama3.2:3b"

results = []

for index, row in df.iterrows():
    text = row['Text (Anonymized)']
    
    # Skip empty rows
    if pd.isna(text):
        continue
    
    text = str(text)
    
    # Get real entity names from columns
    entities = []
    for col in ['Entity_A', 'Entity_B', 'Entity_C', 'Entity_D']:
        if pd.notna(row[col]) and row[col] != '':
            entities.append(str(row[col]))
    
    # Replace placeholders with real names
    real_text = text
    for i, entity in enumerate(entities):
        placeholder = f'[Entity_{"ABCD"[i]}]'
        real_text = real_text.replace(placeholder, entity)
    
    # Run calamanCy on real text
    doc = nlp(real_text)
    detected = [ent.text for ent in doc.ents if ent.label_ == 'PER']
    entity_str = ', '.join(detected) if detected else ', '.join(entities)
    
    prompt = f"""Classify the sentiment of this Taglish political text toward '{entity_str}' as Positive, Negative, or Neutral. Reply with one word only: Positive, Negative, or Neutral.

Text: {real_text}
Sentiment:"""

    response = requests.post(OLLAMA_URL, json={
        "model": MODEL,
        "prompt": prompt,
        "stream": False
    })
    
    prediction = response.json()['response'].strip()
    
    results.append({
        'text': text,
        'real_text': real_text,
        'entity_detected': entity_str,
        'true_label': row['Final_Label'],
        'predicted': prediction
    })
    
    print(f"Row {index}: Entity={entity_str} | Predicted={prediction}")

output = pd.DataFrame(results)
output.to_csv(r'C:\Users\rualryt\OneDrive\Documents\taglish_tosc\results\llama_with_calamancy_stress_entities.csv', index=False)
print("Done!")