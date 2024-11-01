from flask import Flask, render_template, request
import requests
import os

app = Flask(__name__)

# Config
API_URL = 'http://backendapi:5000' 

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/table')
def table():
    try:
        page = request.args.get('page', 1, type=int)
        response = requests.get(f'{API_URL}/predictions', params={'page': page})
        response.raise_for_status() 
        data = response.json()
        
        return render_template('partials/table.html', **data)
    
    except requests.RequestException as e:
        app.logger.error(f"API request failed: {str(e)}")
        return render_template('partials/error.html', 
                             message="Unable to fetch data from API"), 500

@app.route('/predict')
def predict_page():
    return render_template('predict.html')

@app.route('/upload_predict', methods=['POST'])
def upload_predict():
    try:
        if 'file' not in request.files:
            return render_template('partials/prediction_result.html',
                                error="No file uploaded")
        
        file = request.files['file']
        if file.filename == '':
            return render_template('partials/prediction_result.html',
                                error="No file selected")

        if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            return render_template('partials/prediction_result.html',
                                error="Only .jpg, .jpeg, and .png files are allowed")

        # Make prediction request to API
        files = {'file': (file.filename, file.stream, file.content_type)}
        response = requests.post(f'{API_URL}/predict', files=files)
        
        if not response.ok:
            return render_template('partials/prediction_result.html',
                                error=response.json().get('error', 'API Error'))

        result = response.json()
        return render_template('partials/prediction_result.html', **result)

    except Exception as e:
        return render_template('partials/prediction_result.html',
                             error=f"Error: {str(e)}")

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)