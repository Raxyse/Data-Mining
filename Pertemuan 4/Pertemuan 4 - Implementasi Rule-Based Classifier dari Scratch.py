import pandas as pd
import numpy as np

class SimpleRuleBasedClassifier:
    def __init__(self):
        self.rules = []
        self.default_class = None

    def add_rule(self, conditions, class_label, confidence=None):
        """
        Tambah aturan IF-THEN
        conditions: dict {feature: value}
        """
        self.rules.append({
            'conditions': conditions,
            'class': class_label,
            'confidence': confidence or 1.0
        })

        # Urutkan berdasarkan confidence (descending)
        self.rules.sort(key=lambda x: x['confidence'], reverse=True)

    def predict_one(self, x):
        """Prediksi satu data point"""
        for rule in self.rules:
            match = True
            for feature, value in rule['conditions'].items():
                if x[feature] != value:
                    match = False
                    break
            if match:
                return rule['class']
        return self.default_class

    def predict(self, X):
        return [self.predict_one(x) for x in X]

    def set_default(self, class_label):
        self.default_class = class_label


# ======================
# Contoh penggunaan
# ======================

# Dataset sederhana: Outlook, Temperature, Play
data = [
    {'Outlook': 'Sunny', 'Temp': 'Hot', 'Play': 'No'},
    {'Outlook': 'Sunny', 'Temp': 'Mild', 'Play': 'No'},
    {'Outlook': 'Overcast', 'Temp': 'Hot', 'Play': 'Yes'},
    {'Outlook': 'Rainy', 'Temp': 'Mild', 'Play': 'Yes'},
    {'Outlook': 'Rainy', 'Temp': 'Cool', 'Play': 'Yes'}
]

# Buat classifier
clf = SimpleRuleBasedClassifier()

# Tambah aturan
clf.add_rule({'Outlook': 'Overcast'}, 'Yes')
clf.add_rule({'Outlook': 'Sunny', 'Temp': 'Hot'}, 'No', confidence=0.8)
clf.add_rule({'Outlook': 'Rainy'}, 'Yes', confidence=0.7)

# Default class
clf.set_default('Yes')

# Testing
test_data = {'Outlook': 'Sunny', 'Temp': 'Hot'}
print(f"Prediksi: {clf.predict_one(test_data)}")
# Output: No