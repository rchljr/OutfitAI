import os
import numpy as np
from PIL import Image
import tensorflow as tf
from flask import Flask, request, render_template, url_for
from werkzeug.utils import secure_filename

# =============================================================================
# Initialization
# =============================================================================

# Initialize the Flask application
app = Flask(__name__)

# Configure upload folder and allowed file extensions
UPLOAD_FOLDER = 'static/uploads/'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load the pre-trained model
# Ensure the model file name is correct
MODEL_PATH = 'fashion_classifier_improved.h5'
try:
    model = tf.keras.models.load_model(MODEL_PATH)
except IOError:
    print(f"Error: Model file not found at {MODEL_PATH}")
    model = None

# Define the class names in the correct order
# IMPORTANT: Check this order in your notebook with `print(train_gen.class_indices)`
# Example: {'casual': 0, 'formal': 1, 'semi-formal': 2}
# The list must be ['casual', 'formal', 'semi-formal']
CLASS_NAMES = ['casual', 'formal', 'semi-formal'] 


# =============================================================================
# Helper Functions
# =============================================================================

def allowed_file(filename):
    """Checks if the file's extension is allowed."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(image_path, target_size=(224, 224)):
    """
    Loads and preprocesses an image for model prediction.
    """
    try:
        # Open the image
        img = Image.open(image_path)
        
        # Ensure image is in RGB format
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Resize the image
        if img.size != target_size:
            img = img.resize(target_size)
        
        # Convert image to numpy array and normalize
        img_array = np.array(img)
        img_array = np.expand_dims(img_array, axis=0) # Add batch dimension
        img_array = img_array / 255.0
        
        return img_array
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        return None


# =============================================================================
# Flask App Routes
# =============================================================================

@app.route('/', methods=['GET'])
def index():
    """Renders the main page."""
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """Receives an image, predicts its class, and renders the result."""
    if model is None:
        return render_template('index.html', error='Model tidak berhasil dimuat. Cek log server.')

    if 'file' not in request.files:
        return render_template('index.html', error='Tidak ada file yang dipilih.')
    
    file = request.files['file']
    
    if file.filename == '':
        return render_template('index.html', error='Tidak ada file yang dipilih.')

    if file and allowed_file(file.filename):
        # Secure the filename to prevent malicious paths
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Preprocess the image for the model
        processed_image = preprocess_image(filepath)
        if processed_image is None:
            return render_template('index.html', error='Gagal memproses gambar.')

        # Make a prediction
        predictions = model.predict(processed_image)
        
        # Get the result
        predicted_class_index = np.argmax(predictions[0])
        predicted_class_name = CLASS_NAMES[predicted_class_index]
        confidence = np.max(predictions[0]) * 100
        
        # *** IMPORTANT CHANGE HERE ***
        # Use url_for to generate the correct image URL for the template
        image_url = url_for('static', filename='uploads/' + filename)
        
        return render_template(
            'index.html', 
            prediction=predicted_class_name.capitalize(),
            confidence=f"{confidence:.2f}",
            image_path=image_url 
        )
    else:
        return render_template('index.html', error='Format file tidak diizinkan. Gunakan PNG, JPG, atau JPEG.')

# =============================================================================
# Run the App
# =============================================================================
if __name__ == '__main__':
    # Ensure the upload folder exists
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    # Run the app
    app.run(debug=True)

