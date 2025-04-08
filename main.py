import tensorflow as tf
import numpy as np
import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras import layers, models
import io
import base64
import os
import datetime
import plotly.express as px
import plotly.graph_objects as go

# Set page configuration
st.set_page_config(
    page_title="Advanced CIFAR-10 Image Classifier",
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

# CIFAR-10 classes
class_names = ['Plane', 'Car', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

# Check if GPU is available
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

# Load data and preprocess
@st.cache_data
def load_data():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    y_train, y_test = y_train.flatten(), y_test.flatten()
    return (x_train, y_train), (x_test, y_test)

(x_train, y_train), (x_test, y_test) = load_data()

# Build model
def build_model():
    with tf.device('/GPU:0' if physical_devices else '/CPU:0'):
        model = models.Sequential([
            layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(32, 32, 3)),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(10)
        ])
        model.compile(optimizer='adam',
                    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                    metrics=['accuracy'])
    return model

# Train or load model
@st.cache_resource
def get_trained_model(retrain=False, epochs=5):
    model = build_model()
    model_file = 'cifar10_model.h5'
    
    if os.path.exists(model_file) and not retrain:
        model = tf.keras.models.load_model(model_file)
        return model, None
    else:
        with tf.device('/GPU:0' if physical_devices else '/CPU:0'):
            history = model.fit(x_train, y_train, epochs=epochs, 
                              validation_data=(x_test, y_test), verbose=0)
            # Save model
            model.save(model_file)
            return model, history.history

# Sidebar for model settings and visualization options
st.sidebar.image("https://upload.wikimedia.org/wikipedia/en/thumb/2/2e/Cifar-10_1.png/220px-Cifar-10_1.png", 
                use_column_width=True)
st.sidebar.title("Model Settings")
retrain_model = st.sidebar.checkbox("Retrain model")
epochs = st.sidebar.slider("Training epochs", min_value=1, max_value=20, value=5) if retrain_model else 5

with st.spinner("Loading model... This might take a moment."):
    model, history = get_trained_model(retrain_model, epochs)

probability_model = tf.keras.Sequential([model, layers.Softmax()])

st.sidebar.title("Visualization Options")
show_gradcam = st.sidebar.checkbox("Show Grad-CAM visualization", value=True)
show_confusion = st.sidebar.checkbox("Show confusion matrix")
show_examples = st.sidebar.checkbox("Show example images")
show_history = st.sidebar.checkbox("Show training history")

# Collapsible section for educational content
with st.sidebar.expander("📚 How CNNs Work"):
    st.write("Convolutional Neural Networks (CNNs) are specialized neural networks for processing grid-like data such as images.")
    st.write("They use convolutional layers to extract features, pooling layers to reduce dimensions, and fully connected layers for classification.")

# Confusion matrix
def create_confusion_matrix():
    predictions = np.argmax(probability_model.predict(x_test), axis=1)
    cm = tf.math.confusion_matrix(y_test, predictions).numpy()
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm_norm, annot=True, fmt='.2f', xticklabels=class_names, yticklabels=class_names, cmap='Blues')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title('Confusion Matrix')
    return fig

# Grad-CAM implementation
def get_gradcam(img_array, model):
    last_conv_layer = next(l for l in reversed(model.layers) if isinstance(l, layers.Conv2D))
    last_conv_layer_model = tf.keras.Model(model.inputs, last_conv_layer.output)
    
    classifier_input = tf.keras.Input(shape=last_conv_layer.output.shape[1:])
    x = classifier_input
    for layer in model.layers[model.layers.index(last_conv_layer) + 1:]:
        if isinstance(layer, layers.InputLayer):
            continue
        x = layer(x)
    classifier_model = tf.keras.Model(classifier_input, x)
    
    with tf.GradientTape() as tape:
        last_conv_layer_output = last_conv_layer_model(img_array)
        tape.watch(last_conv_layer_output)
        preds = classifier_model(last_conv_layer_output)
        top_pred_index = tf.argmax(preds[0])
        top_class_channel = preds[:, top_pred_index]
    
    grads = tape.gradient(top_class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    last_conv_layer_output = last_conv_layer_output.numpy()[0]
    pooled_grads = pooled_grads.numpy()
    
    for i in range(pooled_grads.shape[-1]):
        last_conv_layer_output[:, :, i] *= pooled_grads[i]
    
    heatmap = np.mean(last_conv_layer_output, axis=-1)
    heatmap = np.maximum(heatmap, 0) / np.max(heatmap)
    
    return heatmap

# Make prediction
def predict(image):
    # Save original dimensions
    orig_image = image.copy()
    
    # Resize for model input
    image = image.resize((32, 32))
    img_array = np.array(image) / 255.0
    img_array_batch = img_array.reshape(1, 32, 32, 3)
    
    with tf.device('/GPU:0' if physical_devices else '/CPU:0'):
        # Get raw predictions
        predictions = probability_model.predict(img_array_batch)
    
    # Get top predictions
    sorted_indices = np.argsort(predictions[0])[::-1]
    sorted_classes = [class_names[i] for i in sorted_indices]
    sorted_confidences = [predictions[0][i] * 100 for i in sorted_indices]
    
    # Generate Grad-CAM if enabled
    overlay = None
    if show_gradcam:
        heatmap = get_gradcam(img_array_batch, model)
        
        # Resize heatmap to original image size
        heatmap = np.uint8(255 * heatmap)
        heatmap = Image.fromarray(heatmap).resize((32, 32), Image.LANCZOS)
        heatmap = np.array(heatmap)
        
        # Apply colormap
        colormap = plt.cm.jet(heatmap)[:, :, :3]
        colormap = np.uint8(255 * colormap)
        
        # Overlay heatmap on original image
        overlaid_img = np.array(image)
        overlay = Image.fromarray(np.uint8(0.6 * overlaid_img + 0.4 * colormap))
    
    return sorted_classes, sorted_confidences, overlay, image

# Function to visualize class examples
@st.cache_data
def get_class_examples():
    examples = {}
    for i, class_name in enumerate(class_names):
        # Get indices for this class
        indices = np.where(y_train == i)[0]
        # Get 5 random examples
        if len(indices) >= 5:
            sample_indices = np.random.choice(indices, 5, replace=False)
            examples[class_name] = x_train[sample_indices]
    
    return examples

# Plot training history with plotly
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
    
    # Plot loss
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=list(range(1, len(history['loss']) + 1)),
        y=history['loss'],
        mode='lines+markers',
        name='Training Loss',
        line=dict(color='royalblue', width=2)
    ))
    
    fig2.add_trace(go.Scatter(
        x=list(range(1, len(history['val_loss']) + 1)),
        y=history['val_loss'],
        mode='lines+markers',
        name='Validation Loss',
        line=dict(color='firebrick', width=2, dash='dash')
    ))
    
    # Update layout
    fig2.update_layout(
        title='Training and Validation Loss',
        xaxis_title='Epoch',
        yaxis_title='Loss',
        legend_title='Legend',
        template='plotly_white'
    )
    
    return fig, fig2

# Create a gauge chart with plotly
def create_gauge_chart(confidence_value):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=confidence_value,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Confidence"},
        gauge={
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
    return fig

# Main app
st.markdown("<h1 class='main-header'>🔍 Advanced CIFAR-10 Image Classifier</h1>", unsafe_allow_html=True)

# Create tabs
tab1, tab2, tab3 = st.tabs(["Image Classification", "Model Performance", "Educational Content"])

with tab1:
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown('<h2 class="sub-header">Upload Image</h2>', unsafe_allow_html=True)
        
        # File uploader
        uploaded_file = st.file_uploader("Drag and drop an image here", 
                                       type=["jpg", "png", "jpeg"],
                                       help="Upload an image of an object to classify")
        
        # Display file size limit
        st.caption("Limit 200MB per file • JPG, PNG, JPEG")
        
        # Or use camera input
        use_camera = st.checkbox("Or use camera input")
        if use_camera:
            camera_image = st.camera_input("Take a picture", label_visibility="collapsed")
            if camera_image:
                uploaded_file = camera_image
    
    with col2:
        st.markdown('<h2 class="sub-header">Results</h2>', unsafe_allow_html=True)
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert("RGB")
            
            # Make prediction
            with st.spinner("Analyzing image..."):
                top_classes, top_confidences, overlay, processed_img = predict(image)
            
            # Display uploaded image
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Display prediction result
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown(f"<h2 style='text-align: center; color: orange;'>{top_classes[0]}</h2>", unsafe_allow_html=True)
            
            # Create gauge chart
            gauge_fig = create_gauge_chart(top_confidences[0])
            st.plotly_chart(gauge_fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Top 3 predictions
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<h3>Top 3 Predictions</h3>', unsafe_allow_html=True)
            
            # Create bar chart using plotly
            top_results = {
                'Class': top_classes[:3],
                'Confidence': top_confidences[:3]
            }
            
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
            
            # Model attention visualization (Grad-CAM)
            if show_gradcam and overlay is not None:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown('<h3>Model Attention (Grad-CAM)</h3>', unsafe_allow_html=True)
                
                col_a, col_b = st.columns(2)
                with col_a:
                    st.image(processed_img, caption="Processed Image", use_column_width=True)
                with col_b:
                    st.image(overlay, caption="Attention Map", use_column_width=True)
                
                st.markdown("<p>The highlighted areas show which parts of the image the model focused on to make its prediction.</p>", unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

with tab2:
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
    
    # Display confusion matrix if enabled
    if show_confusion:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<h3>Confusion Matrix</h3>', unsafe_allow_html=True)
        with st.spinner("Generating confusion matrix..."):
            confusion_fig = create_confusion_matrix()
            st.pyplot(confusion_fig)
        st.markdown("<p>The confusion matrix shows how often the model correctly predicts each class (diagonal) versus how often it confuses classes with each other.</p>", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Display training history if enabled and available
    if show_history and history:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<h3>Training History</h3>', unsafe_allow_html=True)
        
        acc_fig, loss_fig = plot_training_history(history)
        st.plotly_chart(acc_fig, use_container_width=True)
        st.plotly_chart(loss_fig, use_container_width=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    elif show_history and history is None:
        st.info("Training history is not available for pre-trained models. Retrain the model to see training history.")
    
    # Model evaluation
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<h3>Model Evaluation</h3>', unsafe_allow_html=True)
    with st.spinner("Evaluating model..."):
        test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Test Accuracy", f"{test_acc:.2%}")
    with col2:
        st.metric("Test Loss", f"{test_loss:.4f}")
    st.markdown('</div>', unsafe_allow_html=True)

with tab3:
    st.markdown('<h2 class="sub-header">Educational Content</h2>', unsafe_allow_html=True)
    
    # Show example images if enabled
    if show_examples:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<h3>Example Images from CIFAR-10</h3>', unsafe_allow_html=True)
        
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
    
    # Educational text
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<h3>About CIFAR-10</h3>', unsafe_allow_html=True)
    
    st.write("""
    The CIFAR-10 dataset consists of 60,000 32x32 color images in 10 classes, with 6,000 images per class. There are 50,000 training images and 10,000 test images.
    """)
    
    # Use tabs for educational content
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
        
        Our model uses three convolutional layers with filters of increasing complexity to capture different levels of features.
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
        - A dense layer with 128 neurons and ReLU activation
        - Dropout (0.5) to prevent overfitting
        - A final dense layer with 10 neurons (one per class)
        """)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Model architecture visualization
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<h3>How the Model Works</h3>', unsafe_allow_html=True)
    
    st.write("""
    This model uses a Convolutional Neural Network (CNN) architecture:
    
    1. **Convolutional layers** extract features from the input images
    2. **Pooling layers** reduce the spatial dimensions
    3. **Batch normalization** helps stabilize training
    4. **Flatten layer** converts the 2D feature maps to a 1D vector
    5. **Dense layers** perform classification
    6. **Dropout** prevents overfitting
    
    ### Grad-CAM Visualization
    
    Gradient-weighted Class Activation Mapping (Grad-CAM) highlights the regions of the input image that are important for the model's prediction.
    """)
    
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
        label="Conv1+BN";
        color=lightgreen;
        style=filled;
        node [shape=box,style=filled,color=white];
        conv1 [label="32x32x32"];
        pool1 [label="16x16x32"];
      }
      
      subgraph cluster_conv2 {
        label="Conv2+BN";
        color=lightgreen;
        style=filled;
        node [shape=box,style=filled,color=white];
        conv2 [label="16x16x64"];
        pool2 [label="8x8x64"];
      }
      
      subgraph cluster_conv3 {
        label="Conv3+BN";
        color=lightgreen;
        style=filled;
        node [shape=box,style=filled,color=white];
        conv3 [label="8x8x128"];
      }
      
      subgraph cluster_dense {
        label="Dense";
        color=lightyellow;
        style=filled;
        node [shape=box,style=filled,color=white];
        flatten [label="Flatten"];
        dense1 [label="128"];
        dropout [label="Dropout(0.5)"];
        dense2 [label="10"];
      }
      
      input -> conv1 -> pool1 -> conv2 -> pool2 -> conv3 -> flatten -> dense1 -> dropout -> dense2;
    }
    """
    
    st.graphviz_chart(architecture_code)
    st.markdown('</div>', unsafe_allow_html=True)

# Add deploy button in the sidebar
st.sidebar.markdown("---")
st.sidebar.button("Deploy", type="primary")

# Add footer
st.markdown("""
<div style="text-align: center; margin-top: 40px; padding: 20px; background-color: #f0f2f6; border-radius: 10px;">
    <p style="font-size: 14px; color: #666;">CIFAR-10 Image Classifier | Built with TensorFlow and Streamlit</p>
</div>
""", unsafe_allow_html=True)

# If running this script directly
if __name__ == "__main__":
    # Add any additional initialization code here
    pass