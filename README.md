# 🍕 Food-101 Image Classifier

A Vision Transformer (ViT) model fine-tuned on the Food-101 dataset using LoRA (Low-Rank Adaptation) for efficient training. This project includes a Streamlit web application for easy image classification.

## 🚀 Live Demo

[Try it on Streamlit Cloud](#) *(link will be added after deployment)*

## 📋 Features

- **Vision Transformer (ViT)** fine-tuned with LoRA
- **101 Food Categories** classification
- **Interactive Web UI** built with Streamlit
- **Real-time Predictions** with confidence scores
- **Top-K Predictions** display with visual probability bars

## �️ Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
cd YOUR_REPO_NAME
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Download the Model

**Important:** The model file is too large for Git, so it's hosted as a GitHub Release.

1. Go to [Releases](https://github.com/YOUR_USERNAME/YOUR_REPO_NAME/releases)
2. Download `final_model.pkl` from the latest release
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

## � Deploy to Streamlit Cloud

### Prerequisites
1. Push your code to GitHub (without the model file)
2. Upload the model as a GitHub Release (see instructions below)

### Deployment Steps
1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Connect your GitHub repository
3. Set the main file path: `streamlit_app.py`
4. Deploy!

The model will need to be manually downloaded from the release and placed in the `final_model/` folder.

## 📦 Creating a GitHub Release with the Model

### Step-by-Step Instructions:

1. **Commit and Push Your Code** (without model files)
   ```bash
   git add .
   git commit -m "Initial commit: Food-101 classifier"
   git push origin main
   ```

2. **Create a Release on GitHub:**
   - Go to your repository on GitHub
   - Click on "Releases" (right sidebar)
   - Click "Create a new release"
   - Tag version: `v1.0.0`
   - Release title: `Food-101 ViT Model v1.0`
   - Description:
     ```
     Fine-tuned Vision Transformer model for Food-101 classification
     
     Model Details:
     - Base: google/vit-base-patch16-224
     - Method: LoRA fine-tuning
     - Dataset: 20% of Food-101
     - Size: ~340 MB
     
     Download `final_model.pkl` and place it in a `final_model/` folder 
     in the project root directory.
     ```

3. **Upload the Model File:**
   - Drag and drop `final_model/final_model.pkl` to the release assets
   - Click "Publish release"

4. **The model will be available at:**
   ```
   https://github.com/YOUR_USERNAME/YOUR_REPO/releases/download/v1.0.0/final_model.pkl
   ```

## 💡 Usage

1. **Upload an Image:** Click "Choose a food image..." and select a food photo
2. **Classify:** Click the "🔍 Classify Image" button
3. **View Results:** See top predictions with confidence scores

## � Troubleshooting

**Model not found error:**
- Download the model from [Releases](https://github.com/YOUR_USERNAME/YOUR_REPO_NAME/releases)
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
