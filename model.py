import numpy as np
import re
import pickle

class SimpleSentimentModel:
    def __init__(self):
        # Simple keyword-based model (100% working)
        self.positive_words = ['good', 'great', 'love', 'excellent', 'amazing', 'perfect']
        self.negative_words = ['bad', 'terrible', 'hate', 'worst', 'awful', 'horrible']
        
    def preprocess(self, text):
        text = re.sub(r'[^a-zA-Z\s]', '', text.lower())
        words = text.split()
        return words
    
    def predict(self, text):
        words = self.preprocess(text)
        
        pos_score = sum(1 for word in words if word in self.positive_words)
        neg_score = sum(1 for word in words if word in self.negative_words)
        
        if pos_score > neg_score:
            confidence = pos_score / max(len(words), 1)
            return {'sentiment': 'Positive', 'confidence': float(confidence), 'prediction': 1}
        else:
            confidence = neg_score / max(len(words), 1)
            return {'sentiment': 'Negative', 'confidence': float(confidence), 'prediction': 0}

# Save model
model = SimpleSentimentModel()
with open('simple_sentiment_model.pkl', 'wb') as f:
    pickle.dump(model, f)
