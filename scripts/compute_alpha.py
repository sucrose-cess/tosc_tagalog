import pandas as pd
import krippendorff
import numpy as np

gold = pd.read_csv(r'C:\Users\rualryt\OneDrive\Documents\taglish_tosc\data\gold_dataset.csv')
stress = pd.read_csv(r'C:\Users\rualryt\OneDrive\Documents\taglish_tosc\data\stress_dataset.csv')

mapping = {'Negative': -1, 'Neutral': 0, 'Positive': 1}

for name, df in [('Gold', gold), ('Stress', stress)]:
    r1 = df['R1'].map(mapping)
    r2 = df['R2'].map(mapping)
    r3 = df['R3'].map(mapping)
    
    matrix = np.array([r1.tolist(), r2.tolist(), r3.tolist()])
    
    alpha = krippendorff.alpha(reliability_data=matrix, level_of_measurement='ordinal')
    
    print(f"{name} dataset Krippendorff's Alpha: {round(alpha, 4)}")

print("\nDone!")