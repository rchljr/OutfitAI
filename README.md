# OutfitAI: AI-Powered Fashion Style Classifier

OutfitAI is a lightweight and intuitive web-based application that utilizes deep learning to classify clothing style from images. Simply upload a photo, and the model will tell you whether the outfit is Casual, Formal, or Semi-Formal — all within seconds.

This project highlights an end-to-end deep learning pipeline, from data labeling and preprocessing to model training and deployment with Flask.

![OutfitAI Web Application Screenshot](./assets/Screenshot.png)

---

## 🧠 How It Works

- A fine-tuned ResNet50 model analyzes uploaded outfit images.
- The model outputs one of three fashion categories: Casual, Formal, or Semi-Formal.
- A confidence score (in percentage) is displayed for each prediction.
- The web UI is built with Tailwind CSS and Flask for an elegant and fast user experience.

---

## 🗂️ Dataset

We used a curated subset of the DeepFashion-MultiModal dataset from Kaggle:

📦 Dataset: DeepFashion-MultiModal  
🔗 Source: https://www.kaggle.com/datasets/silverstone1903/deep-fashion-multimodal  
🧷 Description: Contains diverse fashion images and metadata including category, gender, and style attributes. For this project, we filtered samples by product_type and manually mapped them into 3 style classes: Casual, Formal, and Semi-Formal.

Example label mapping:

- Casual: T-Shirts, Hoodies, Denim
- Formal: Suits, Blazers, Dress Shirts
- Semi-Formal: Cardigans, Skirts, Sweaters

---

## ✨ Features

✅ Upload and classify fashion photos in real-time  
✅ Visual feedback with predicted class and model confidence  
✅ Built using TensorFlow + Keras with ResNet50 backbone  
✅ Works on desktop browsers (Chrome, Firefox, Edge)  
✅ Clean, modern UI using Tailwind CSS  

---

## 🔥 Example Output

When a user uploads this image:

🖼️ ![example-input](./assets/example-input.jpg)

The application might output:

> ✅ Predicted Style: Semi-Formal  
> 📊 Confidence Scores:
> - Casual: 12.4%
> - Formal: 25.1%
> - Semi-Formal: 62.5%

---

## 🛠️ Tech Stack

- Backend: Python, Flask
- Deep Learning: TensorFlow, Keras, Scikit-learn
- Frontend: HTML5, Tailwind CSS
- Image Handling: Pillow
- Version Control: Git + Git LFS (for large model handling)

---

## 🚀 Getting Started

Follow the steps below to run the project locally.

### Prerequisites

- Python 3.10+
- Git & Git LFS installed

### Installation

1. Clone this repository:

    ```bash
    git clone https://github.com/rchljr/OutfitAI.git
    cd OutfitAI
    ```

2. Create and activate a virtual environment:

    ```bash
    # Windows
    py -m venv .venv
    .\\.venv\\Scripts\\activate

    # macOS / Linux
    python3 -m venv .venv
    source .venv/bin/activate
    ```

3. Install dependencies:

    ```bash
    pip install -r requirements.txt
    ```

4. Pull the model file using Git LFS:

    ```bash
    git lfs pull
    ```

    This will download fashion_classifier_improved.h5

---

## ▶️ Running the App

1. Start the Flask development server:

    ```bash
    flask run
    ```

2. Open your browser and visit: http://127.0.0.1:5000

---

## 📉 Current Accuracy

- Validation Accuracy: 44%
- Model: Fine-tuned ResNet50 on 3-class custom dataset
- Optimizer: Adam (lr=1e-4)
- Loss Function: Categorical Crossentropy

Confusion matrix shows misclassification is most common between Formal and Semi-Formal categories — reflecting the inherent ambiguity in style perception.

---

## 🔭 Future Improvements

- Simplify task to binary classification (e.g. Casual vs. Formal) for higher accuracy
- Improve dataset balance and class separation
- Test newer architectures (e.g. MobileNetV3, EfficientNetV2)
- Integrate more context-aware inputs like full-body segmentation

---

🧵 This project was built as a portfolio piece and experiment in fashion + AI. If you like it, feel free to fork, suggest improvements, or reach out!
"""

# Write to a markdown file
readme_path = "/mnt/data/README_OutfitAI.md"
with open(readme_path, "w", encoding="utf-8") as f:
    f.write(readme_content)

readme_path
