from flask import Flask, request, jsonify, current_app
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import logging
from datetime import datetime
import io
import redis
import base64
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
r = redis.Redis(host='localhost', port=6379, decode_responses=True)

# Load model at startup
try:
    model = load_model('best_model.keras')
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Error loading model: {str(e)}")
    raise

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy", "timestamp": str(datetime.now())})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Check if file was uploaded
        if 'file' not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files['file']
        
        # Basic file validation
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400
            
        if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            return jsonify({"error": "Invalid file type"}), 400

        # Read file once
        file_content = file.read()
        
        # Create base64
        encoded_image = base64.b64encode(file_content).decode('utf-8')
        
        # Create BytesIO for image processing
        file_bytes = io.BytesIO(file_content)
        
        # Process image
        img = image.load_img(file_bytes, target_size=(224, 224))
        img_array = image.img_to_array(img) / 255.
        img_array = np.expand_dims(img_array, axis=0)
        
        # Make prediction
        prediction = model.predict(img_array)
        confidence = float(prediction[0][0])
        result = "Pneumonia" if confidence > 0.5 else "Normal"
        
        # Log prediction
        logger.info(f"Prediction made for {file.filename}: {result} ({confidence:.2%})")
        
        # Return result
        return jsonify({
            "prediction": result,
            "confidence": f"{confidence:.2%}",
            "filename": file.filename,
            "image": encoded_image,
            "timestamp": str(datetime.now())
        })

    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/predictions', methods=['GET'])
def get_predictions():
    try:
        page = request.args.get('page', 1, type=int)
        if page < 1:
            return jsonify({'error': 'Page must be >= 1'}), 400

        ITEMS_PER_PAGE = 25

        keys = r.keys('*')
        total_keys = len(keys)
        
        start = (page - 1) * ITEMS_PER_PAGE
        end = start + ITEMS_PER_PAGE
        
        if start >= total_keys and total_keys > 0:
            return jsonify({'error': 'Page out of range'}), 400
            
        keys_to_display = keys[start:end]

        data = {}
        for key in keys_to_display:
            try:
                # No need to decode since decode_responses=True
                data[key] = r.hgetall(key)
            except Exception as e:
                app.logger.error(f"Error processing key {key}: {str(e)}")
                continue

        return jsonify({
            'data': data,
            'page': page,
            'total_keys': total_keys,
            'items_per_page': ITEMS_PER_PAGE,
            'total_pages': (total_keys // ITEMS_PER_PAGE) + (1 if total_keys % ITEMS_PER_PAGE > 0 else 0)
        })

    except redis.RedisError as e:
        app.logger.error(f"Redis error: {str(e)}")  # Use app.logger instead of current_app
        return jsonify({'error': 'Database error'}), 500
    except Exception as e:
        app.logger.error(f"Unexpected error: {str(e)}")  # Use app.logger instead of current_app
        return jsonify({'error': 'Server error'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)