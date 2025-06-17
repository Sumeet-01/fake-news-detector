from flask import Flask, render_template, request
import pickle
import os

# Load model and vectorizer
model = pickle.load(open("model/fake_news_model.pkl", "rb"))
vectorizer = pickle.load(open("model/tfidf_vectorizer.pkl", "rb"))

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        news_text = request.form['news']
        vect_text = vectorizer.transform([news_text])
        prediction = model.predict(vect_text)[0]
        label = "REAL" if prediction == 1 else "FAKE"
        return render_template('result.html', prediction=label, news=news_text)

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
