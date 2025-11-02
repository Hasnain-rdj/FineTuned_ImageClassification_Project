"""
Streamlit Application for Food-101 Image Classification
Loads the fine-tuned ViT model and provides an interactive interface for image upload and prediction.
"""

import streamlit as st
import torch
import pickle
import numpy as np
from PIL import Image
from pathlib import Path
import sys

# Set page configuration
st.set_page_config(
    page_title="Food-101 Classifier",
    page_icon="🍕",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #FF6B6B;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #4ECDC4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .stProgress > div > div > div > div {
        background-color: #FF6B6B;
    }
    </style>
""", unsafe_allow_html=True)

def download_model_from_release():
    """Download model from GitHub release if not present locally."""
    model_dir = Path(__file__).parent / 'final_model'
    model_path = model_dir / 'final_model.pkl'
    
    if model_path.exists():
        return model_path
    
    # Create directory if it doesn't exist
    model_dir.mkdir(exist_ok=True)
    
    # GitHub release information
    GITHUB_REPO = "Hasnain-rdj/FineTuned_ImageClassification_Project"
    RELEASE_TAG = "v1.0.0"
    MODEL_FILENAME = "final_model.pkl"
    release_url = f"https://github.com/{GITHUB_REPO}/releases/download/{RELEASE_TAG}/{MODEL_FILENAME}"
    
    st.info(f"📥 Downloading model from GitHub Release {RELEASE_TAG}...")
    st.info(f"File size: ~330 MB (this may take 2-5 minutes depending on your connection)")
    
    try:
        import urllib.request
        
        # Progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Download with progress tracking
        def reporthook(block_num, block_size, total_size):
            downloaded = block_num * block_size
            percent = min(int(downloaded * 100 / total_size), 100)
            progress_bar.progress(percent)
            
            # Convert bytes to MB
            downloaded_mb = downloaded / (1024 * 1024)
            total_mb = total_size / (1024 * 1024)
            status_text.text(f"Downloaded: {downloaded_mb:.1f} MB / {total_mb:.1f} MB ({percent}%)")
        
        # Download the file
        urllib.request.urlretrieve(release_url, model_path, reporthook)
        
        # Clear progress indicators
        progress_bar.empty()
        status_text.empty()
        
        st.success(f"✅ Model downloaded successfully! ({model_path.stat().st_size / (1024**2):.1f} MB)")
        return model_path
    
    except Exception as e:
        st.error(f"❌ Failed to download model: {str(e)}")
        st.error(f"Please download manually from: https://github.com/{GITHUB_REPO}/releases/tag/{RELEASE_TAG}")
        st.info("After downloading, place the file in: final_model/final_model.pkl")
        st.stop()

@st.cache_resource
def load_model():
    """Load the fine-tuned ViT model from the final_model directory."""
    model_path = download_model_from_release()
    
    if not model_path.exists():
        st.error(f"Model file not found at {model_path}")
        st.stop()
    
    try:
        # Set device first - force CPU for local machine
        device = torch.device('cpu')
        
        # Load model data with CPU mapping
        import warnings
        warnings.filterwarnings('ignore')
        
        # Use pickle with custom unpickler to handle device issues
        import io
        
        class CPUUnpickler(pickle.Unpickler):
            def find_class(self, module, name):
                if module == 'torch.storage' and name == '_load_from_bytes':
                    return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
                else:
                    return super().find_class(module, name)
        
        with open(model_path, 'rb') as f:
            model_data = CPUUnpickler(f).load()
        
        # Extract model components
        id2label = model_data['id2label']
        label2id = model_data['label2id']
        classes = model_data['classes']
        test_accuracy = model_data.get('test_accuracy', 0.0)
        
        # Recreate processor from scratch instead of using saved one
        from transformers import ViTImageProcessor
        processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
        
        # Reconstruct the model architecture
        from transformers import ViTForImageClassification
        from peft import LoraConfig, get_peft_model
        
        model = ViTForImageClassification.from_pretrained(
            'google/vit-base-patch16-224',
            num_labels=len(classes),
            id2label=id2label,
            label2id=label2id,
            ignore_mismatched_sizes=True
        )
        
        # Apply LoRA configuration
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=['query', 'value'],
            lora_dropout=0.1,
            bias='none',
            modules_to_save=['classifier']
        )
        
        model = get_peft_model(model, lora_config)
        
        # Move state dict tensors to CPU if needed
        state_dict = model_data['model_state']
        cpu_state_dict = {}
        for key, value in state_dict.items():
            if isinstance(value, torch.Tensor):
                cpu_state_dict[key] = value.cpu()
            else:
                cpu_state_dict[key] = value
        
        model.load_state_dict(cpu_state_dict, strict=False)
        
        model.eval()
        model.to(device)
        
        return model, processor, id2label, classes, test_accuracy, device
    
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        import traceback
        st.error(traceback.format_exc())
        st.stop()

def predict_image(image, model, processor, id2label, device, top_k=5):
    """
    Predict the class of a food image.
    
    Args:
        image: PIL Image
        model: Fine-tuned ViT model
        processor: Image processor
        id2label: Dictionary mapping indices to labels
        device: torch device
        top_k: Number of top predictions to return
    
    Returns:
        List of dictionaries with class names and probabilities
    """
    # Preprocess the image
    inputs = processor(images=image, return_tensors='pt')
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Get predictions
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = torch.nn.functional.softmax(logits, dim=-1)[0]
    
    # Get top-k predictions
    top_probs, top_indices = torch.topk(probabilities, top_k)
    
    results = []
    for prob, idx in zip(top_probs, top_indices):
        class_name = id2label[idx.item()]
        results.append({
            'class': class_name,
            'probability': prob.item()
        })
    
    return results

def main():
    # Header
    st.markdown('<h1 class="main-header">🍕 Food-101 Image Classifier</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">Upload an image of food and get instant predictions using our fine-tuned Vision Transformer model!</p>',
        unsafe_allow_html=True
    )
    
    # Load model with temporary status messages
    loading_placeholder = st.empty()
    
    with loading_placeholder.container():
        with st.spinner('🔄 Loading model... This may take a moment on first run.'):
            model, processor, id2label, classes, test_accuracy, device = load_model()
        
        # Show success message
        success_msg = st.success("✅ Model loaded successfully!")
        
        # Auto-hide success message after 15 seconds
        import time
        time.sleep(15)
        success_msg.empty()
    
    # Clear the loading placeholder to clean the dashboard
    loading_placeholder.empty()
    
    # Sidebar
    with st.sidebar:
        st.header("ℹ️ About")
        st.info(
            f"""
            This app uses a **Vision Transformer (ViT)** model fine-tuned on the Food-101 dataset 
            using **LoRA** (Low-Rank Adaptation) for efficient training.
            
            **Model Details:**
            - Base Model: `google/vit-base-patch16-224`
            - Number of Classes: {len(classes)}
            - Device: {device.type.upper()}
            
            **Fine-tuning Method:**
            - LoRA Rank: 16
            - Training: FP16 precision
            - Dataset: 20% subset of Food-101
            """
        )
        
        st.header("📊 Model Settings")
        top_k = st.slider("Number of top predictions to show", 1, 10, 5)
        show_probabilities = st.checkbox("Show probability bars", value=True)
        
        st.header("📚 Food Categories")
        with st.expander("View all 101 food categories"):
            for i, category in enumerate(sorted(classes), 1):
                st.text(f"{i}. {category.replace('_', ' ').title()}")
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📤 Upload Image")
        uploaded_file = st.file_uploader(
            "Choose a food image...",
            type=['jpg', 'jpeg', 'png'],
            help="Upload an image of food for classification"
        )
        
        if uploaded_file is not None:
            # Display uploaded image
            image = Image.open(uploaded_file).convert('RGB')
            st.image(image, caption='Uploaded Image', use_container_width=True)
            
            # Add predict button
            if st.button("🔍 Classify Image", type="primary", use_container_width=True):
                with st.spinner('Analyzing image...'):
                    predictions = predict_image(image, model, processor, id2label, device, top_k)
                    st.session_state['predictions'] = predictions
    
    with col2:
        st.header("🎯 Predictions")
        
        if 'predictions' in st.session_state:
            predictions = st.session_state['predictions']
            
            # Display top prediction prominently
            top_pred = predictions[0]
            st.success(f"**Top Prediction: {top_pred['class'].replace('_', ' ').title()}**")
            st.metric(
                label="Confidence",
                value=f"{top_pred['probability']*100:.2f}%"
            )
            
            # Display all predictions
            st.markdown("### All Predictions")
            
            for i, pred in enumerate(predictions, 1):
                class_name = pred['class'].replace('_', ' ').title()
                probability = pred['probability']
                
                # Create columns for ranking
                pred_col1, pred_col2 = st.columns([3, 1])
                
                with pred_col1:
                    st.markdown(f"**{i}. {class_name}**")
                    if show_probabilities:
                        st.progress(probability)
                
                with pred_col2:
                    st.markdown(f"**{probability*100:.2f}%**")
        else:
            st.info("👆 Upload an image and click 'Classify Image' to see predictions!")
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666;'>
            <p>Built with ❤️ using Streamlit | Model: Vision Transformer + LoRA | Dataset: Food-101</p>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
