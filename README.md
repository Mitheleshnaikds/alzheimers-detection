# Alzheimer's Disease Detection System

A deep learning-powered web application for classifying Alzheimer's disease stages from brain MRI images.

## 🎯 Overview

This project provides an interactive web interface for detecting and classifying Alzheimer's disease from MRI scans using a Convolutional Neural Network (CNN). The system achieves **91.86% accuracy** in identifying four stages of cognitive impairment.

## ✨ Features

- 🧠 **AI-Powered Classification**: Deep learning CNN model trained on 33,984 MRI images
- 🎨 **Modern Web Interface**: Beautiful purple gradient UI built with Streamlit
- 📊 **Real-time Predictions**: Upload MRI images and get instant classification results
- 📈 **Confidence Scores**: View prediction probabilities for all disease stages
- 🚀 **Fast Processing**: Optimized inference pipeline for quick results

## 🏥 Disease Stages Detected

1. **Non Demented** - No signs of cognitive impairment
2. **Very Mild Demented** - Early-stage symptoms
3. **Mild Demented** - Moderate progression
4. **Moderate Demented** - Advanced symptoms

## 🛠️ Technologies Used

- **TensorFlow/Keras** - Deep learning framework for model training and inference
- **Streamlit** - Interactive web application framework
- **Python 3.x** - Core programming language
- **NumPy** - Numerical computing
- **Pillow** - Image processing
- **Matplotlib** - Data visualization

## 📁 Project Structure

```
├── App/
│   ├── streamlit_app.py    # Web application interface
│   ├── model.h5            # Trained CNN model (91.86% accuracy)
│   └── requirements.txt    # Python dependencies
├── dataset/
│   └── AugmentedAlzheimerDataset/  # Training data (33,984 images)
├── train_model.py          # Model training script
├── README.md
└── LICENSE
```

## 🚀 Getting Started

### Prerequisites

- Python 3.9 or higher
- Virtual environment (recommended)

### Installation

1. **Clone the repository**
```bash
git clone <your-repo-url>
cd alzheimers-detection-main
```

2. **Create and activate virtual environment**
```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Mac/Linux
```

3. **Install dependencies**
```bash
cd App
pip install -r requirements.txt
```

### Running the Application

```bash
C:/Users/mithe/Desktop/alzheimers-detection-main/alzheimers-detection-main/.venv/Scripts/python.exe -m streamlit run streamlit_app.py

streamlit run streamlit_app.py
```

The app will open in your browser at `http://localhost:8502`

## 🎓 Training Your Own Model

If you want to retrain the model with your own dataset:

1. **Prepare your dataset**
   - Place MRI images in `dataset/AugmentedAlzheimerDataset/`
   - Organize into 4 class folders: `MildDemented`, `ModerateDemented`, `NonDemented`, `VeryMildDemented`

2. **Run training script**
```bash
python train_model.py
```

3. **Training details**
   - Image size: 128x128 pixels
   - Batch size: 32 (CPU) or 64 (GPU)
   - Epochs: 100 (with early stopping)
   - Data augmentation: rotation, shifting, zooming, flipping

## 🏗️ Model Architecture

Custom CNN with the following layers:
- 4 Convolutional layers (32 → 64 → 64 → 128 filters)
- MaxPooling after each conv layer
- Flatten layer
- 2 Dense layers (256 → 128 neurons)
- Dropout (0.5) for regularization
- Output layer: 4 classes (softmax)

**Total Parameters**: 1,343,492

## 📊 Model Performance

- **Test Accuracy**: 91.86%
- **Training Images**: 27,188
- **Validation Images**: 6,796
- **Callbacks**: Early stopping, learning rate reduction, model checkpointing

## 🎨 Web Interface Features

- Clean, modern purple gradient design
- Drag-and-drop image upload
- Real-time image preview
- Classification results with confidence percentages
- Visual confidence breakdown for all classes
- Stage descriptions with emoji indicators
- Responsive layout

## 📝 Usage Example

1. Open the web app in your browser
2. Upload an MRI scan image (JPG, PNG, or JPEG)
3. View the uploaded image preview
4. See the predicted disease stage
5. Check confidence scores for all stages

## 🔮 Future Improvements

- Support for DICOM medical image format
- Batch processing for multiple images
- Export results to PDF reports
- Integration with medical databases
- Model explanation visualizations (Grad-CAM)
- Support for additional disease stages

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## ⚠️ Disclaimer

This tool is for educational and research purposes only. It should not be used as a substitute for professional medical diagnosis. Always consult with qualified healthcare professionals for medical decisions.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📧 Contact

For questions or feedback, please open an issue on GitHub.
