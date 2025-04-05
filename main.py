import tensorflow as tf
import numpy as np
import streamlit as st
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io
import cv2
import time
import os
from tensorflow.keras import layers, models
from tensorflow.keras.models import load_model
import datetime
import plotly.express as px
import plotly.graph_objects as go
from streamlit_image_comparison import image_comparison

# Set page configuration
st.set_page_config(
    page_title="CIFAR-10 Image Classifier",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem !important;
        font-weight: 700 !important;
        color: #1E88E5 !important;
        text-align: center !important;
        margin-bottom: 2rem !important;
    }
    .sub-header {
        font-size: 1.5rem !important;
        font-weight: 600 !important;
        color: #26A69A !important;
        margin-top: 1.5rem !important;
    }
    .card {
        padding: 1.5rem;
        border-radius: 0.5rem;
        background-color: #f8f9fa;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
    }
    .metric-value {
        font-size: 1.8rem !important;
        font-weight: 700 !important;
        color: #303F9F !important;
    }
    .metric-label {
        font-size: 1rem !important;
        color: #616161 !important;
    }
    .stProgress .st-bo {
        background-color: #4CAF50 !important;
    }
</style>
""", unsafe_allow_html=True)

# Enable GPU memory growth
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    try:
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)
        st.sidebar.success(f"✅ Using GPU acceleration ({len(physical_devices)} GPU(s) detected)")
    except Exception as e:
        st.sidebar.warning(f"⚠️ Error configuring GPU: {e}")
else:
    st.sidebar.info("💻 No GPU found. Using CPU.")

# CIFAR-10 classes
class_names = ['Plane', 'Car', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

# Function to create and train model
@st.cache_resource
def create_and_train_model(retrain=False, epochs=5):
    # Check if saved model exists
    if os.path.exists('cifar10_model.h5') and not retrain:
        return load_model('cifar10_model.h5'), None
    
    # Load data and preprocess
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    y_train, y_test = y_train.flatten(), y_test.flatten()
    
    # Build model
    with tf.device('/GPU:0' if physical_devices else '/CPU:0'):
        model = models.Sequential([
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.Flatten(),
            layers.Dense(64, activation='relu'),
            layers.Dense(10)
        ])
        model.compile(optimizer='adam',
                    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                    metrics=['accuracy'])
        
        # Train model with history
        history = model.fit(x_train, y_train, epochs=epochs, 
                          validation_data=(x_test, y_test), verbose=0)
        
        # Save model
        model.save('cifar10_model.h5')
        
        return model, history.history

# Function to make prediction
def predict(model, image):
    # Original image dimensions
    orig_width, orig_height = image.size
    
    # Preprocessing for visualization
    processed_img = image.resize((32, 32))
    
    # Convert to array for prediction
    img_array = np.array(processed_img) / 255.0
    img_array = img_array.reshape(1, 32, 32, 3)
    
    # Use GPU for prediction if available
    with tf.device('/GPU:0' if physical_devices else '/CPU:0'):
        # Get raw logits
        prediction_logits = model.predict(img_array)
        # Convert to probabilities
        prediction_probs = tf.nn.softmax(prediction_logits).numpy()[0]
    
    # Get all class predictions sorted
    sorted_indices = np.argsort(prediction_probs)[::-1]
    sorted_probs = prediction_probs[sorted_indices]
    sorted_classes = [class_names[i] for i in sorted_indices]
    
    return sorted_classes, sorted_probs, processed_img

# Function to visualize class examples
@st.cache_data
def get_class_examples():
    # Load data
    (_, _), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    
    examples = {}
    for i, class_name in enumerate(class_names):
        # Get indices for this class
        indices = np.where(y_test == i)[0]
        # Get 5 random examples
        if len(indices) >= 5:
            sample_indices = np.random.choice(indices, 5, replace=False)
            examples[class_name] = x_test[sample_indices]
    
    return examples

# Function to generate Grad-CAM heatmap
def generate_gradcam(model, img_array, class_idx):
    # Create model for grad-cam
    grad_model = tf.keras.models.Model(
        [model.inputs], 
        [model.get_layer('conv2d_2').output, model.output]
    )
    
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, class_idx]
    
    # Extract gradients
    grads = tape.gradient(loss, conv_outputs)
    
    # Global average pooling
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    # Weight the channels by the gradients
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)
    
    # Normalize the heatmap
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    heatmap = heatmap.numpy()
    
    # Resize heatmap to original image size
    heatmap = cv2.resize(heatmap, (32, 32))
    
    # Convert heatmap to RGB
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    
    # Superimpose heatmap on original image
    img_bgr = cv2.cvtColor(np.uint8(img_array[0] * 255), cv2.COLOR_RGB2BGR)
    superimposed_img = heatmap * 0.4 + img_bgr
    superimposed_img = np.clip(superimposed_img, 0, 255).astype('uint8')
    
    return cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB)

# Function to plot training history
def plot_training_history(history):
    fig = go.Figure()
    
    # Plot accuracy
    fig.add_trace(go.Scatter(
        x=list(range(1, len(history['accuracy']) + 1)),
        y=history['accuracy'],
        mode='lines+markers',
        name='Training Accuracy',
        line=dict(color='royalblue', width=2)
    ))
    
    fig.add_trace(go.Scatter(
        x=list(range(1, len(history['val_accuracy']) + 1)),
        y=history['val_accuracy'],
        mode='lines+markers',
        name='Validation Accuracy',
        line=dict(color='firebrick', width=2, dash='dash')
    ))
    
    # Update layout
    fig.update_layout(
        title='Training and Validation Accuracy',
        xaxis_title='Epoch',
        yaxis_title='Accuracy',
        legend_title='Legend',
        template='plotly_white'
    )
    
    return fig

# Function to plot confusion matrix
@st.cache_data
def plot_confusion_matrix():
    # Load data
    (_, _), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    x_test = x_test / 255.0
    y_test = y_test.flatten()
    
    # Load model
    model = load_model('cifar10_model.h5')
    
    # Get predictions
    with tf.device('/GPU:0' if physical_devices else '/CPU:0'):
        predictions = model.predict(x_test[:1000])  # Use a subset for speed
    
    # Convert to class indices
    pred_classes = np.argmax(predictions, axis=1)
    
    # Create confusion matrix
    cm = tf.math.confusion_matrix(y_test[:1000], pred_classes).numpy()
    
    # Plot using seaborn
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    
    # Convert plot to image
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    
    return buf

# Main application
st.markdown('<h1 class="main-header">🔍 Advanced CIFAR-10 Image Classifier</h1>', unsafe_allow_html=True)

# Sidebar
st.sidebar.image("https://upload.wikimedia.org/wikipedia/en/thumb/2/2e/Cifar-10_1.png/220px-Cifar-10_1.png", 
                use_column_width=True)
st.sidebar.markdown("## Model Settings")

# Model training options
retrain = st.sidebar.checkbox("Retrain model", value=False)
epochs = st.sidebar.slider("Training epochs", min_value=1, max_value=20, value=5) if retrain else 5

# Load/train model
with st.spinner("Loading model... This might take a moment."):
    model, history = create_and_train_model(retrain, epochs)

# Create probability model
probability_model = tf.keras.Sequential([model, layers.Softmax()])

# Add visualization options
st.sidebar.markdown("## Visualization Options")
show_gradcam = st.sidebar.checkbox("Show Grad-CAM visualization", value=True)
show_conf_matrix = st.sidebar.checkbox("Show confusion matrix", value=False)
show_examples = st.sidebar.checkbox("Show example images", value=False)
show_history = st.sidebar.checkbox("Show training history", value=False)

# Educational section
with st.sidebar.expander("📚 How CNNs Work"):
    st.markdown("""
    Convolutional Neural Networks (CNNs) consist of:
    1. **Convolutional layers** extract features like edges and shapes
    2. **Pooling layers** reduce dimensions while preserving features
    3. **Fully connected layers** make final predictions
    
    This model analyzes patterns in images to classify them into 10 categories.
    """)

# Main content
tabs = st.tabs(["Image Classification", "Model Performance", "Educational Content"])

# Image Classification Tab
with tabs[0]:
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown('<h2 class="sub-header">Upload Image</h2>', unsafe_allow_html=True)
        
        # File uploader with drag and drop
        uploaded_file = st.file_uploader(
            "Drag and drop an image here", 
            type=["jpg", "png", "jpeg"],
            help="Upload an image of an object to classify"
        )
        
        # Camera input option
        use_camera = st.checkbox("Or use camera input")
        if use_camera:
            camera_image = st.camera_input("Take a picture")
            if camera_image:
                uploaded_file = camera_image
    
    # Process uploaded image
    if uploaded_file is not None:
        # Open image
        image = Image.open(uploaded_file).convert("RGB")
        
        # Make prediction
        with st.spinner("Analyzing image..."):
            sorted_classes, sorted_probs, processed_img = predict(model, image)
        
        # Display images side by side
        with col1:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.image(image, caption="Original Image", use_column_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<h2 class="sub-header">Results</h2>', unsafe_allow_html=True)
            
            # Display top prediction
            st.markdown(f'<div class="card"><h3>Top Prediction: <span style="color:#FF5722;">{sorted_classes[0]}</span></h3>', unsafe_allow_html=True)
            
            # Create confidence gauge
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = sorted_probs[0] * 100,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Confidence"},
                gauge = {
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "#FF5722"},
                    'steps': [
                        {'range': [0, 50], 'color': "#FFECB3"},
                        {'range': [50, 80], 'color': "#FFD54F"},
                        {'range': [80, 100], 'color': "#FFC107"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 80
                    }
                }
            ))
            
            fig.update_layout(height=200, margin=dict(l=20, r=20, t=0, b=0))
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Display top 3 predictions
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<h3>Top 3 Predictions</h3>', unsafe_allow_html=True)
            
            # Create DataFrame for top predictions
            top_results = pd.DataFrame({
                'Class': sorted_classes[:3],
                'Confidence': sorted_probs[:3] * 100
            })
            
            # Display horizontal bar chart
            fig = px.bar(
                top_results, 
                x='Confidence', 
                y='Class', 
                orientation='h',
                text='Confidence',
                color='Confidence',
                color_continuous_scale='Blues',
                text_auto='.1f'
            )
            fig.update_layout(
                xaxis_title="Confidence (%)",
                yaxis_title="",
                height=200,
                margin=dict(l=20, r=20, t=10, b=30)
            )
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Grad-CAM visualization
            if show_gradcam:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown('<h3>Model Attention (Grad-CAM)</h3>', unsafe_allow_html=True)
                
                # Processed image for Grad-CAM
                img_array = np.array(processed_img) / 255.0
                img_array = img_array.reshape(1, 32, 32, 3)
                
                # Generate Grad-CAM for top prediction
                top_class_idx = np.argmax(sorted_probs)
                heatmap = generate_gradcam(model, img_array, top_class_idx)
                
                # Show original and heatmap side by side
                col_a, col_b = st.columns(2)
                with col_a:
                    st.image(processed_img, caption="Processed Image", use_column_width=True)
                with col_b:
                    st.image(heatmap, caption="Attention Map", use_column_width=True)
                
                st.markdown("<p>The highlighted areas show which parts of the image the model focused on to make its prediction.</p>", unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

# Model Performance Tab
with tabs[1]:
    st.markdown('<h2 class="sub-header">Model Performance</h2>', unsafe_allow_html=True)
    
    # Display metrics in cards
    metric_cols = st.columns(4)
    with metric_cols[0]:
        st.markdown('<div class="card" style="text-align: center;">', unsafe_allow_html=True)
        st.markdown('<p class="metric-label">Model Architecture</p>', unsafe_allow_html=True)
        st.markdown('<p class="metric-value">CNN</p>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with metric_cols[1]:
        st.markdown('<div class="card" style="text-align: center;">', unsafe_allow_html=True)
        st.markdown('<p class="metric-label">Parameters</p>', unsafe_allow_html=True)
        st.markdown(f'<p class="metric-value">{model.count_params():,}</p>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with metric_cols[2]:
        st.markdown('<div class="card" style="text-align: center;">', unsafe_allow_html=True)
        st.markdown('<p class="metric-label">Classes</p>', unsafe_allow_html=True)
        st.markdown('<p class="metric-value">10</p>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with metric_cols[3]:
        st.markdown('<div class="card" style="text-align: center;">', unsafe_allow_html=True)
        st.markdown('<p class="metric-label">Last Updated</p>', unsafe_allow_html=True)
        if os.path.exists('cifar10_model.h5'):
            mod_time = os.path.getmtime('cifar10_model.h5')
            date_str = datetime.datetime.fromtimestamp(mod_time).strftime('%Y-%m-%d')
            st.markdown(f'<p class="metric-value">{date_str}</p>', unsafe_allow_html=True)
        else:
            st.markdown('<p class="metric-value">New</p>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Display confusion matrix
    if show_conf_matrix:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<h3>Confusion Matrix</h3>', unsafe_allow_html=True)
        
        with st.spinner("Generating confusion matrix..."):
            cm_image = plot_confusion_matrix()
            st.image(cm_image, caption="Confusion Matrix", use_column_width=True)
        
        st.markdown("<p>The confusion matrix shows how often the model correctly predicts each class (diagonal) versus how often it confuses classes with each other.</p>", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Display training history
    if show_history and history is not None:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<h3>Training History</h3>', unsafe_allow_html=True)
        
        history_fig = plot_training_history(history)
        st.plotly_chart(history_fig, use_container_width=True)
        
        st.markdown("<p>The graph shows how model accuracy improved during training.</p>", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    elif show_history and history is None:
        st.info("Training history is not available for pre-trained models. Retrain the model to see training history.")

# Educational Content Tab
with tabs[2]:
    st.markdown('<h2 class="sub-header">Learn About CIFAR-10</h2>', unsafe_allow_html=True)
    
    # Class examples
    if show_examples:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<h3>Example Images from Each Class</h3>', unsafe_allow_html=True)
        
        with st.spinner("Loading example images..."):
            examples = get_class_examples()
            
            for i, class_name in enumerate(class_names):
                if i % 2 == 0:
                    cols = st.columns(2)
                
                with cols[i % 2]:
                    st.write(f"**{class_name}**")
                    images = examples[class_name]
                    image_cols = st.columns(5)
                    for j, img in enumerate(images):
                        with image_cols[j]:
                            st.image(img, width=50)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Educational content about CNNs
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<h3>How Convolutional Neural Networks Work</h3>', unsafe_allow_html=True)
    
    cnn_tabs = st.tabs(["Overview", "Convolutional Layers", "Pooling", "Classification"])
    
    with cnn_tabs[0]:
        st.markdown("""
        **Convolutional Neural Networks (CNNs)** are a specialized type of neural networks designed for processing structured grid data such as images.
        
        The key innovation of CNNs is their ability to automatically learn spatial hierarchies of features through:
        - **Convolutional Layers**: Extract features like edges, textures, and patterns
        - **Pooling Layers**: Reduce dimensionality while maintaining important information
        - **Fully Connected Layers**: Combine these features for final classification
        
        For our CIFAR-10 classification task, the model learns to identify visual patterns that distinguish between the 10 object categories.
        """)
    
    with cnn_tabs[1]:
        st.markdown("""
        **Convolutional layers** are the core building blocks of CNNs.
        
        How they work:
        1. Small filters (kernels) slide across the input image
        2. Each filter performs element-wise multiplication and summing
        3. Different filters detect different features (edges, corners, textures)
        
        Our model uses three convolutional layers with 32, 64, and 64 filters respectively, each with a 3×3 kernel size.
        """)
        
        st.image("https://miro.medium.com/max/1400/1*ciDgQEjViWLnCbmX-EeSrA.gif", caption="Convolution operation")
        
    with cnn_tabs[2]:
        st.markdown("""
        **Pooling layers** reduce the spatial dimensions of the data.
        
        Benefits:
        - Reduces computation
        - Controls overfitting
        - Creates position invariance
        
        Our model uses Max Pooling with 2×2 filters, which takes the maximum value in each 2×2 region.
        """)
        
        st.image("https://miro.medium.com/max/790/1*uoWYsCV5vBU8SHFPAPao-w.gif", caption="Max pooling operation")
        
    with cnn_tabs[3]:
        st.markdown("""
        After feature extraction, the CNN uses **fully connected layers** for classification:
        
        1. Features are flattened into a 1D vector
        2. Dense layers learn non-linear combinations of these features
        3. The final layer has 10 neurons (one for each CIFAR-10 class)
        
        Our model uses:
        - A flatten layer to convert 2D feature maps to 1D
        - A dense layer with 64 neurons and ReLU activation
        - A final dense layer with 10 neurons (one per class)
        """)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Model architecture visualization
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<h3>Model Architecture</h3>', unsafe_allow_html=True)
    
    # Simplified architecture diagram
    architecture_code = """
    digraph G {
      rankdir=LR;
      splines=line;
      
      subgraph cluster_input {
        label="Input";
        color=lightblue;
        style=filled;
        node [shape=box,style=filled,color=white];
        input [label="32x32x3"];
      }
      
      subgraph cluster_conv1 {
        label="Conv1";
        color=lightgreen;
        style=filled;
        node [shape=box,style=filled,color=white];
        conv1 [label="32x32x32"];
        pool1 [label="16x16x32"];
      }
      
      subgraph cluster_conv2 {
        label="Conv2";
        color=lightgreen;
        style=filled;
        node [shape=box,style=filled,color=white];
        conv2 [label="16x16x64"];
        pool2 [label="8x8x64"];
      }
      
      subgraph cluster_conv3 {
        label="Conv3";
        color=lightgreen;
        style=filled;
        node [shape=box,style=filled,color=white];
        conv3 [label="8x8x64"];
      }
      
      subgraph cluster_dense {
        label="Dense";
        color=lightyellow;
        style=filled;
        node [shape=box,style=filled,color=white];
        flatten [label="4096"];
        dense1 [label="64"];
        dense2 [label="10"];
      }
      
      input -> conv1 -> pool1 -> conv2 -> pool2 -> conv3 -> flatten -> dense1 -> dense2;
    }
    """
    
    st.graphviz_chart(architecture_code)
    st.markdown('</div>', unsafe_allow_html=True)

# Add footer
st.markdown("""
<div style="text-align: center; margin-top: 40px; padding: 20px; background-color: #f0f2f6; border-radius: 10px;">
    <p style="font-size: 14px; color: #666;">CIFAR-10 Image Classifier | Built with TensorFlow and Streamlit</p>
</div>
""", unsafe_allow_html=True)

# Requirements.txt should include:
# tensorflow
# numpy
# streamlit
# pillow
# opencv-python
# pandas
# matplotlib
# seaborn
# plotly
# streamlit-image-comparison