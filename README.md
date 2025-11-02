# 🍕 Food-101 Image Classifier

A Vision Transformer (ViT) model fine-tuned on the Food-101 dataset using LoRA (Low-Rank Adaptation) for efficient training. This project includes a Streamlit web application for easy image classification.

## 🚀 Live Demo

[Try it on Streamlit Cloud](https://imgclassifierai.streamlit.app/)

## 📋 Features

- **Vision Transformer (ViT)** fine-tuned with LoRA
- **101 Food Categories** classification
- **Interactive Web UI** built with Streamlit
- **Real-time Predictions** with confidence scores
- **Top-K Predictions** display with visual probability bars

## �️ Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/Hasnain-rdj/FineTuned_ImageClassification_Project.git
cd FineTuned_ImageClassification_Project
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Download the Model

**Good News!** The app will automatically download the model from GitHub Release on first run.

Alternatively, you can manually download:

1. Go to [Releases](https://github.com/Hasnain-rdj/FineTuned_ImageClassification_Project/releases)
2. Download `final_model.pkl` from the latest release (v1.0.0)
3. Create a `final_model` folder in the project directory
4. Place `final_model.pkl` inside the `final_model` folder

```
Task3/
├── final_model/
│   └── final_model.pkl  <- Place the downloaded file here
├── streamlit_app.py
├── requirements.txt
└── ...
```

### 4. Run the Application

```bash
streamlit run streamlit_app.py
```

The app will open in your browser at `http://localhost:8501`

## � Model Details

- **Base Model:** `google/vit-base-patch16-224`
- **Fine-tuning Method:** LoRA (Low-Rank Adaptation)
- **Dataset:** Food-101 (20% subset)
- **Number of Classes:** 101 food categories
- **Training Configuration:**
  - LoRA Rank: 16
  - LoRA Alpha: 32
  - Learning Rate: 2e-4
  - Batch Size: 32
  - Epochs: 5
  - Precision: FP16

## 🍽️ Food Categories

The model can classify 101 different food categories including:
- Pizza, Sushi, Tacos, Hamburger
- Apple Pie, Cheesecake, Chocolate Cake
- Spaghetti Carbonara, Pad Thai, Ramen
- And 92 more delicious categories!

## 📁 Project Structure

```
Task3/
├── streamlit_app.py          # Main Streamlit application
├── vit_food101_finetuning.ipynb  # Training notebook
├── requirements.txt          # Python dependencies
├── final_model/              # Model files (from GitHub Release)
│   └── final_model.pkl
└── README.md                 # This file
```

## 🌐 Deploy to Streamlit Cloud

### Quick Deploy Steps

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Sign in with your GitHub account
3. Click **"New app"**
4. Fill in:
   - **Repository**: `Hasnain-rdj/FineTuned_ImageClassification_Project`
   - **Branch**: `main`
   - **Main file path**: `streamlit_app.py`
5. Click **"Deploy!"**

The app will automatically download the model from your GitHub release on first launch!

## 📦 GitHub Release Info

The trained model is available as a GitHub Release to keep the repository size small.

- **Release URL**: https://github.com/Hasnain-rdj/FineTuned_ImageClassification_Project/releases/tag/v1.0.0
- **Model File**: `final_model.pkl` (~340 MB)
- **Auto-download**: The Streamlit app automatically downloads the model on first run

## 💡 Usage

1. **Upload an Image:** Click "Choose a food image..." and select a food photo
2. **Classify:** Click the "🔍 Classify Image" button
3. **View Results:** See top predictions with confidence scores

## 🔧 Troubleshooting

**Model download fails:**
- Check your internet connection
- Manually download from [Releases](https://github.com/Hasnain-rdj/FineTuned_ImageClassification_Project/releases)
- Place it in the `final_model/` folder

**Out of memory:**
- The app runs on CPU by default
- Ensure you have at least 2GB of free RAM

**Slow predictions:**
- First prediction is slower due to model loading (cached afterward)
- CPU inference takes 2-5 seconds per image

## 📄 License

This project uses the Food-101 dataset and pre-trained models from Hugging Face.

## 🙏 Acknowledgments

- Food-101 Dataset
- Hugging Face Transformers
- Microsoft PEFT (LoRA)
- Streamlit

## 📧 Contact

For questions or issues, please open an issue on GitHub.

---

Built with ❤️ using Vision Transformer + LoRA
