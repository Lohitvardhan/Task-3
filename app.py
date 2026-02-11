from flask import Flask, render_template_string, request, jsonify
import re

app = Flask(__name__)

class ProfessionalSentimentModel:
    def __init__(self):
        # Expanded + weighted keyword lists
        self.positive_words = {
            'love': 3, 'excellent': 3, 'perfect': 3, 'amazing': 3, 'fantastic': 3,
            'great': 2, 'good': 2, 'awesome': 2, 'wonderful': 2, 'superb': 2,
            'best': 2, 'happy': 2, 'recommend': 2
        }
        self.negative_words = {
            'hate': 3, 'terrible': 3, 'awful': 3, 'horrible': 3, 'worst': 3,
            'bad': 2, 'poor': 2, 'disappointing': 2, 'trash': 2, 'sucks': 2,
            'terrible': 2, 'awful': 2, 'hate': 2
        }
    
    def preprocess(self, text):
        # Clean + extract words
        text = re.sub(r'[^a-zA-Z\s]', '', text.lower())
        words = [w for w in text.split() if len(w) > 2]
        return words
    
    def predict(self, text):
        words = self.preprocess(text)
        if not words:
            return {'sentiment': 'Neutral', 'confidence': 0.5, 'prediction': 0}
        
        pos_score = sum(self.positive_words.get(word, 0) for word in words)
        neg_score = sum(self.negative_words.get(word, 0) for word in words)
        
        total_score = pos_score - neg_score
        total_words = len(words)
        
        # Calculate confidence (0.7-1.0 range for professional feel)
        if total_score > 0:
            sentiment = 'Positive'
            base_conf = min(pos_score / total_words * 1.5, 1.0)
        elif total_score < 0:
            sentiment = 'Negative' 
            base_conf = min(neg_score / total_words * 1.5, 1.0)
        else:
            sentiment = 'Neutral'
            base_conf = 0.5
        
        # Boost confidence for strong signals
        confidence = max(base_conf, 0.75) if abs(total_score) > 1 else base_conf
        
        return {
            'sentiment': sentiment,
            'confidence': round(float(confidence), 2),
            'prediction': 1 if sentiment == 'Positive' else 0,
            'pos_score': float(pos_score),
            'neg_score': float(neg_score)
        }

model = ProfessionalSentimentModel()

@app.route('/')
def home():
    return '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Professional Sentiment Analysis - Task 3</title>
        <style>
            body { font-family: -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto; max-width: 900px; margin: 50px auto; padding: 20px; background: #f8f9fa; }
            .container { background: white; padding: 40px; border-radius: 15px; box-shadow: 0 10px 30px rgba(0,0,0,0.1); }
            h1 { color: #2c3e50; text-align: center; margin-bottom: 10px; }
            .subtitle { text-align: center; color: #7f8c8d; margin-bottom: 30px; }
            textarea { width: 100%; height: 140px; padding: 15px; border: 2px solid #e1e8ed; border-radius: 10px; font-size: 16px; box-sizing: border-box; resize: vertical; }
            button { background: linear-gradient(45deg, #4CAF50, #45a049); color: white; padding: 15px 40px; border: none; border-radius: 10px; cursor: pointer; font-size: 18px; font-weight: bold; transition: transform 0.2s; }
            button:hover { transform: translateY(-2px); }
            .result { margin-top: 25px; padding: 25px; border-radius: 12px; font-size: 18px; font-weight: bold; box-shadow: 0 5px 15px rgba(0,0,0,0.1); }
            .positive { background: linear-gradient(135deg, #d4edda, #c3e6cb); color: #155724; border-left: 6px solid #28a745; }
            .negative { background: linear-gradient(135deg, #f8d7da, #f5c6cb); color: #721c24; border-left: 6px solid #dc3545; }
            .neutral { background: linear-gradient(135deg, #e2e3e5, #d6d8db); color: #495057; border-left: 6px solid #6c757d; }
            .metrics { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin-top: 15px; font-size: 16px; }
            .metric { background: rgba(255,255,255,0.7); padding: 12px; border-radius: 8px; text-align: center; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🎯 Professional Sentiment Analysis</h1>
             <p class="subtitle"> (Deployed API)</p>
            <textarea id="textInput" placeholder="Enter your review... e.g., 'I love this amazing product!' or 'This is absolutely terrible...'"></textarea><br><br>
            <button onclick="analyze()">🔍 Analyze Sentiment</button>
            <div id="result"></div>
        </div>
        <script>
            async function analyze() {
                const text = document.getElementById('textInput').value.trim();
                if (!text) return alert('Please enter some text!');
                
                document.getElementById('result').innerHTML = '<div style="text-align:center;padding:20px;">Analyzing...</div>';
                
                try {
                    const response = await fetch('/predict', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify({text: text})
                    });
                    const result = await response.json();
                    displayResult(result);
                } catch (error) {
                    document.getElementById('result').innerHTML = '<div style="color:red;padding:20px;">Error analyzing text</div>';
                }
            }
            
            function displayResult(result) {
                const resultDiv = document.getElementById('result');
                const className = result.sentiment.toLowerCase();
                resultDiv.innerHTML = `
                    <div class="result ${className}">
                        <div style="font-size:24px;margin-bottom:10px;">${result.sentiment}</div>
                        <div class="metrics">
                            <div class="metric">
                                <strong>Confidence</strong><br>
                                ${(result.confidence * 100).toFixed(1)}%
                            </div>
                            <div class="metric">
                                <strong>Prediction</strong><br>
                                ${result.prediction}
                            </div>
                            <div class="metric">
                                <strong>Positive Score</strong><br>
                                ${result.pos_score.toFixed(1)}
                            </div>
                            <div class="metric">
                                <strong>Negative Score</strong><br>
                                ${result.neg_score.toFixed(1)}
                            </div>
                        </div>
                    </div>
                `;
            }
        </script>
    </body>
    </html>
    '''

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    text = data.get('text', '')
    result = model.predict(text)
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
