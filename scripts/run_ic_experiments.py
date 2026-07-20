import pandas as pd
import requests

templates = pd.read_csv(r'C:\Users\rualryt\OneDrive\Documents\taglish_tosc\data\sentence_templates.csv')
entities = pd.read_csv(r'C:\Users\rualryt\OneDrive\Documents\taglish_tosc\data\entity.csv.csv')


OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "llama3.2:3b"

results = []

for _, entity_row in entities.iterrows():
    entity_id = entity_row['Entity_ID']
    entity_name = entity_row['Official name']
    order_axis = entity_row['Order_Axis']
    social_axis = entity_row['Social_Axis']
    temporal_axis = entity_row['Temporal_Axis']

    for _, template_row in templates.iterrows():
        template_id = template_row['template_id']
        template_text = template_row['template_text']
        sentiment_type = template_row['sentiment_type']

        real_text = template_text.replace('[ENTITY]', entity_name)

        prompt = f"""Classify the sentiment of this Taglish political text toward '{entity_name}' as Positive, Negative, or Neutral. Reply with one word only: Positive, Negative, or Neutral.

Text: {real_text}
Sentiment:"""

        response = requests.post(OLLAMA_URL, json={
            "model": MODEL,
            "prompt": prompt,
            "stream": False
        })

        prediction = response.json()['response'].strip()

        results.append({
            'template_id': template_id,
            'template_text': template_text,
            'template_sentiment': sentiment_type,
            'entity_id': entity_id,
            'entity_name': entity_name,
            'order_axis': order_axis,
            'social_axis': social_axis,
            'temporal_axis': temporal_axis,
            'real_text': real_text,
            'predicted': prediction
        })

        print(f"Template {template_id} | Entity: {entity_name} | Predicted: {prediction}")

output = pd.DataFrame(results)
output.to_csv(r'C:\Users\rualryt\OneDrive\Documents\taglish_tosc\results\llama_ic_results.csv', index=False)
print("Done!")