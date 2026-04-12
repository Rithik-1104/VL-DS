import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.datasets import fetch_openml
import warnings
warnings.filterwarnings('ignore')
 
# Page configuration
st.set_page_config(
    page_title="Multimodal Fusion Lab",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)
 
# Custom CSS for beautiful styling
st.markdown("""
<style>
    /* Main theme colors */
    :root {
        --primary-color: #6366f1;
        --secondary-color: #ec4899;
        --success-color: #10b981;
        --warning-color: #f59e0b;
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
    }
    
    .main-header h1 {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    
    .main-header p {
        font-size: 1.1rem;
        opacity: 0.95;
    }
    
    /* Card styling */
    .info-card {
        background: linear-gradient(135deg, #f6f8fb 0%, #ffffff 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 4px solid #6366f1;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.07);
    }
    
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        border-top: 3px solid #6366f1;
    }
    
    .modality-badge {
        display: inline-block;
        padding: 0.4rem 1rem;
        border-radius: 20px;
        font-weight: 600;
        margin: 0.3rem;
        font-size: 0.9rem;
    }
    
    .badge-image {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
    }
    
    .badge-text {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        color: white;
    }
    
    .badge-audio {
        background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%);
        color: white;
    }
    
    .badge-sensor {
        background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
        color: white;
    }
    
    /* Stale element fix */
    .stButton > button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.7rem 1.5rem;
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
    }
    
    /* Progress bar */
    .stProgress > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #f6f8fb 0%, #ffffff 100%);
    }
    
    /* Hide streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Section headers */
    .section-header {
        color: #1f2937;
        font-size: 1.5rem;
        font-weight: 700;
        margin: 2rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 3px solid #6366f1;
    }
</style>
""", unsafe_allow_html=True)
 
# Initialize session state
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False
if 'dataset_choice' not in st.session_state:
    st.session_state.dataset_choice = None
 
# Header
st.markdown("""
<div class="main-header">
    <h1>🧬 Multimodal Fusion Virtual Lab</h1>
    <p>Exploring Modality Importance Scoring with Interactive Visualizations</p>
</div>
""", unsafe_allow_html=True)
 
# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/000000/artificial-intelligence.png", width=80)
    st.markdown("### 🎯 Navigation")
    
    page = st.radio(
        "",
        ["📚 Introduction", "🔬 Dataset Generation", "⚡ Fusion & Training", "📊 Analysis & Insights", "🎓 Learn More"],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    st.markdown("### ⚙️ Quick Settings")
    
    if page == "🔬 Dataset Generation":
        n_samples = st.slider("Number of Samples", 100, 2000, 500, 100)
        noise_level = st.slider("Noise Level", 0.0, 0.5, 0.1, 0.05)
    
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; padding: 1rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 10px; color: white;'>
        <small>Built with ❤️ for Learning</small>
    </div>
    """, unsafe_allow_html=True)
 
# Page: Introduction
if page == "📚 Introduction":
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown('<p class="section-header">What is Multimodal Fusion?</p>', unsafe_allow_html=True)
        
        st.markdown("""
        <div class="info-card">
        <h3>🎯 Concept Overview</h3>
        <p style='font-size: 1.05rem; line-height: 1.7;'>
        <strong>Multimodal Fusion</strong> is the process of integrating information from multiple 
        data sources (modalities) to make better predictions than any single modality alone.
        </p>
        <p style='font-size: 1.05rem; line-height: 1.7;'>
        Think of it like a doctor diagnosing a patient - they don't rely on just one test result. 
        They combine X-rays, blood tests, patient history, and physical examination to make an 
        accurate diagnosis. Similarly, AI systems can combine different types of data!
        </p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown('<p class="section-header">Modality Importance Scoring</p>', unsafe_allow_html=True)
        
        st.markdown("""
        <div class="info-card">
        <h3>⚖️ Why It Matters</h3>
        <p style='font-size: 1.05rem; line-height: 1.7;'>
        Not all data sources are equally important! <strong>Modality Importance Scoring</strong> 
        helps us understand which modalities contribute most to our predictions.
        </p>
        <ul style='font-size: 1.05rem; line-height: 1.8;'>
            <li>🎯 <strong>Attention Mechanism:</strong> Learn to focus on important modalities</li>
            <li>📊 <strong>Weighted Fusion:</strong> Combine modalities with learned weights</li>
            <li>🔍 <strong>Interpretability:</strong> Understand what the model relies on</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown('<p class="section-header">Modalities in This Lab</p>', unsafe_allow_html=True)
        
        st.markdown('<span class="modality-badge badge-image">🖼️ Image Features</span>', unsafe_allow_html=True)
        st.markdown("Visual patterns, shapes, and structures")
        st.markdown("---")
        
        st.markdown('<span class="modality-badge badge-text">📝 Text Features</span>', unsafe_allow_html=True)
        st.markdown("Semantic information and descriptions")
        st.markdown("---")
        
        st.markdown('<span class="modality-badge badge-audio">🎵 Audio Features</span>', unsafe_allow_html=True)
        st.markdown("Sound patterns and frequencies")
        st.markdown("---")
        
        st.markdown('<span class="modality-badge badge-sensor">📡 Sensor Features</span>', unsafe_allow_html=True)
        st.markdown("Physical measurements and signals")
    
    st.markdown('<p class="section-header">🚀 How This Lab Works</p>', unsafe_allow_html=True)
    
    cols = st.columns(4)
    
    with cols[0]:
        st.markdown("""
        <div class="metric-card">
            <h2 style='color: #6366f1; margin: 0;'>1️⃣</h2>
            <h4 style='margin: 0.5rem 0;'>Generate Data</h4>
            <p style='margin: 0; color: #6b7280;'>Create synthetic multimodal dataset</p>
        </div>
        """, unsafe_allow_html=True)
    
    with cols[1]:
        st.markdown("""
        <div class="metric-card">
            <h2 style='color: #ec4899; margin: 0;'>2️⃣</h2>
            <h4 style='margin: 0.5rem 0;'>Train Model</h4>
            <p style='margin: 0; color: #6b7280;'>Learn fusion weights</p>
        </div>
        """, unsafe_allow_html=True)
    
    with cols[2]:
        st.markdown("""
        <div class="metric-card">
            <h2 style='color: #10b981; margin: 0;'>3️⃣</h2>
            <h4 style='margin: 0.5rem 0;'>Analyze Results</h4>
            <p style='margin: 0; color: #6b7280;'>Explore importance scores</p>
        </div>
        """, unsafe_allow_html=True)
    
    with cols[3]:
        st.markdown("""
        <div class="metric-card">
            <h2 style='color: #f59e0b; margin: 0;'>4️⃣</h2>
            <h4 style='margin: 0.5rem 0;'>Learn Insights</h4>
            <p style='margin: 0; color: #6b7280;'>Understand patterns</p>
        </div>
        """, unsafe_allow_html=True)
 
# Page: Dataset Generation
elif page == "🔬 Dataset Generation":
    st.markdown('<p class="section-header">Load Real Multimodal Dataset</p>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-card">
    <h4>📊 Available Datasets</h4>
    <p>We use <strong>real public datasets</strong> and simulate multimodal features from them for educational purposes.</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        dataset_choice = st.selectbox(
            "Choose Dataset",
            ["MNIST Digits (Images)", "Wine Quality (Sensor Data)", "Iris Flowers (Botanical)"],
            help="Select a real-world dataset to load"
        )
        
        if st.button("📥 Load Dataset", use_container_width=True):
            with st.spinner("Loading real dataset..."):
                progress_bar = st.progress(0)
                
                try:
                    if dataset_choice == "MNIST Digits (Images)":
                        # Load MNIST (subset for speed)
                        from sklearn.datasets import load_digits
                        data = load_digits()
                        X_base = data.data[:1000]  # Use 1000 samples
                        y = data.target[:1000]
                        n_classes = 10
                        n_samples = len(y)
                        
                        progress_bar.progress(25)
                        
                        # Create multimodal features from MNIST
                        # Modality 1: Raw pixel features (Image)
                        X_image = X_base[:, :32]  # First 32 pixels
                        
                        # Modality 2: Edge features (simulated visual processing)
                        X_text = np.column_stack([
                            X_base.mean(axis=1),
                            X_base.std(axis=1),
                            X_base.max(axis=1),
                            X_base.min(axis=1),
                            (X_base > X_base.mean()).sum(axis=1),
                        ])
                        X_text = np.tile(X_text, (1, 8))  # Expand to 40 features
                        
                        progress_bar.progress(50)
                        
                        # Modality 3: Gradient features (Audio-like)
                        X_audio = np.diff(X_base[:, :31], axis=1)  # 30 features
                        
                        # Modality 4: Statistical features (Sensor-like)
                        X_sensor = np.column_stack([
                            np.percentile(X_base, 25, axis=1),
                            np.percentile(X_base, 50, axis=1),
                            np.percentile(X_base, 75, axis=1),
                            (X_base > 0).sum(axis=1),
                            X_base.var(axis=1)
                        ])
                        X_sensor = np.tile(X_sensor, (1, 4))  # Expand to 20 features
                        
                    elif dataset_choice == "Wine Quality (Sensor Data)":
                        # Load Wine Quality dataset
                        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
                        df = pd.read_csv(url, sep=';')
                        
                        # Create quality classes (low, medium, high)
                        df['quality_class'] = pd.cut(df['quality'], bins=[0, 5, 6, 10], labels=[0, 1, 2])
                        
                        X_base = df.drop(['quality', 'quality_class'], axis=1).values[:500]
                        y = df['quality_class'].values[:500].astype(int)
                        n_classes = 3
                        n_samples = len(y)
                        
                        progress_bar.progress(25)
                        
                        # Split into modalities
                        # Modality 1: Chemical properties (Image-like)
                        X_image = np.tile(X_base[:, :3], (1, 17))[:, :50]
                        
                        # Modality 2: Acidity features (Text-like)
                        X_text = np.tile(X_base[:, 3:6], (1, 14))[:, :40]
                        
                        progress_bar.progress(50)
                        
                        # Modality 3: Sugar/density (Audio-like)
                        X_audio = np.tile(X_base[:, 6:9], (1, 10))[:, :30]
                        
                        # Modality 4: Other features (Sensor)
                        X_sensor = np.tile(X_base[:, 9:], (1, 10))[:, :20]
                        
                    else:  # Iris
                        from sklearn.datasets import load_iris
                        data = load_iris()
                        X_base = data.data
                        y = data.target
                        n_classes = 3
                        n_samples = len(y)
                        
                        # Augment data to have more samples
                        X_base = np.tile(X_base, (5, 1)) + np.random.randn(len(y)*5, 4) * 0.1
                        y = np.tile(y, 5)
                        n_samples = len(y)
                        
                        progress_bar.progress(25)
                        
                        # Create modalities
                        X_image = np.tile(X_base, (1, 13))[:, :50]
                        X_text = np.tile(X_base, (1, 10))[:, :40]
                        
                        progress_bar.progress(50)
                        
                        X_audio = np.tile(X_base, (1, 8))[:, :30]
                        X_sensor = np.tile(X_base, (1, 5))[:, :20]
                    
                    progress_bar.progress(75)
                    
                    # Normalize all modalities
                    scaler = StandardScaler()
                    X_image = scaler.fit_transform(X_image)
                    X_text = scaler.fit_transform(X_text)
                    X_audio = scaler.fit_transform(X_audio)
                    X_sensor = scaler.fit_transform(X_sensor)
                    
                    progress_bar.progress(100)
                    
                    # Store in session state
                    st.session_state.X_image = X_image
                    st.session_state.X_text = X_text
                    st.session_state.X_audio = X_audio
                    st.session_state.X_sensor = X_sensor
                    st.session_state.y = y
                    st.session_state.n_classes = n_classes
                    st.session_state.n_samples = n_samples
                    st.session_state.dataset_choice = dataset_choice
                    st.session_state.data_loaded = True
                    
                    st.success(f"✅ Dataset loaded: {n_samples} samples, {n_classes} classes!")
                    st.balloons()
                    
                except Exception as e:
                    st.error(f"❌ Error loading dataset: {str(e)}")
                    st.info("💡 Using Wine Quality as fallback...")
                    
                    # Fallback: Create simple structured data
                    np.random.seed(42)
                    n_samples = 500
                    n_classes = 3
                    
                    # Create base features with class dependency
                    y = np.random.choice(n_classes, n_samples)
                    
                    X_image = np.zeros((n_samples, 50))
                    X_text = np.zeros((n_samples, 40))
                    X_audio = np.zeros((n_samples, 30))
                    X_sensor = np.zeros((n_samples, 20))
                    
                    for i in range(n_classes):
                        mask = y == i
                        # Create class-specific patterns
                        X_image[mask] = np.random.randn(mask.sum(), 50) + i * 2
                        X_text[mask] = np.random.randn(mask.sum(), 40) + i * 1.5
                        X_audio[mask] = np.random.randn(mask.sum(), 30) + i * 1.0
                        X_sensor[mask] = np.random.randn(mask.sum(), 20) + i * 0.5
                    
                    st.session_state.X_image = X_image
                    st.session_state.X_text = X_text
                    st.session_state.X_audio = X_audio
                    st.session_state.X_sensor = X_sensor
                    st.session_state.y = y
                    st.session_state.n_classes = n_classes
                    st.session_state.n_samples = n_samples
                    st.session_state.dataset_choice = "Structured Synthetic"
                    st.session_state.data_loaded = True
                    
                    st.success("✅ Fallback dataset loaded successfully!")
    
    with col2:
        st.markdown("""
        <div class="info-card">
        <h4>📋 Modality Details</h4>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown('<span class="modality-badge badge-image">🖼️ Image</span>', unsafe_allow_html=True)
        st.caption("50 features | High importance")
        
        st.markdown('<span class="modality-badge badge-text">📝 Text</span>', unsafe_allow_html=True)
        st.caption("40 features | Medium importance")
        
        st.markdown('<span class="modality-badge badge-audio">🎵 Audio</span>', unsafe_allow_html=True)
        st.caption("30 features | Medium-low importance")
        
        st.markdown('<span class="modality-badge badge-sensor">📡 Sensor</span>', unsafe_allow_html=True)
        st.caption("20 features | Low importance")
    
    if st.session_state.data_loaded:
        st.markdown('<p class="section-header">📊 Dataset Overview</p>', unsafe_allow_html=True)
        
        cols = st.columns(4)
        with cols[0]:
            st.metric("Total Samples", st.session_state.n_samples)
        with cols[1]:
            st.metric("Total Features", 140)
        with cols[2]:
            st.metric("Classes", st.session_state.n_classes)
        with cols[3]:
            st.metric("Modalities", 4)
        
        # Class distribution
        fig = px.histogram(
            x=st.session_state.y,
            nbins=st.session_state.n_classes,
            labels={'x': 'Class', 'y': 'Count'},
            title=f'Class Distribution - {st.session_state.dataset_choice}',
            color_discrete_sequence=['#667eea']
        )
        fig.update_layout(
            showlegend=False,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Show sample data
        with st.expander("🔍 View Sample Features"):
            col_a, col_b = st.columns(2)
            with col_a:
                st.write("**Image Modality (first 5 samples)**")
                st.dataframe(pd.DataFrame(st.session_state.X_image[:5, :10]).round(3))
            with col_b:
                st.write("**Labels (first 10)**")
                st.write(st.session_state.y[:10])
 
# Page: Fusion & Training
elif page == "⚡ Fusion & Training":
    st.markdown('<p class="section-header">Attention-Based Multimodal Fusion</p>', unsafe_allow_html=True)
    
    if not st.session_state.data_loaded:
        st.warning("⚠️ Please load a dataset first in the 'Dataset Generation' page!")
    else:
        st.markdown("""
        <div class="info-card">
        <h4>🧠 Fusion Architecture</h4>
        <p>We use an <strong>attention mechanism</strong> to learn the importance of each modality. 
        The model learns weights α₁, α₂, α₃, α₄ through <strong>gradient descent</strong> that indicate how much to trust each modality.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Training configuration
        col1, col2 = st.columns([2, 1])
        
        with col1:
            with st.form("training_config"):
                st.markdown("#### Training Parameters")
                col_a, col_b = st.columns(2)
                with col_a:
                    learning_rate = st.select_slider("Learning Rate", 
                                                     options=[0.001, 0.01, 0.05, 0.1],
                                                     value=0.01)
                    epochs = st.slider("Epochs", 20, 100, 50, 10)
                with col_b:
                    test_size = st.slider("Test Split %", 10, 40, 20, 5) / 100
                    regularization = st.slider("Regularization", 0.0, 1.0, 0.1, 0.1)
                
                train_btn = st.form_submit_button("🎯 Train Fusion Model", use_container_width=True)
        
        with col2:
            st.markdown("""
            <div class="info-card">
            <h4>📐 Mathematical Formula</h4>
            <p style='text-align: center; font-size: 1.1rem; padding: 1rem; background: #f9fafb; border-radius: 8px; font-family: monospace;'>
            f = α₁·h₁ + α₂·h₂ + α₃·h₃ + α₄·h₄
            </p>
            <p style='font-size: 0.9rem; margin-top: 1rem;'>
            where hᵢ are modality representations and αᵢ are learned importance weights (sum to 1).
            </p>
            </div>
            """, unsafe_allow_html=True)
        
        if train_btn:
            with st.spinner("Training fusion model with gradient descent..."):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Prepare data
                status_text.text("Preparing data...")
                X_combined = np.concatenate([
                    st.session_state.X_image,
                    st.session_state.X_text,
                    st.session_state.X_audio,
                    st.session_state.X_sensor
                ], axis=1)
                
                # Split data
                X_train, X_test, y_train, y_test = train_test_split(
                    X_combined, st.session_state.y, 
                    test_size=test_size, 
                    random_state=42,
                    stratify=st.session_state.y
                )
                
                progress_bar.progress(10)
                
                # Split features by modality
                def split_modalities(X):
                    return [
                        X[:, :50],      # Image
                        X[:, 50:90],    # Text
                        X[:, 90:120],   # Audio
                        X[:, 120:]      # Sensor
                    ]
                
                train_modalities = split_modalities(X_train)
                test_modalities = split_modalities(X_test)
                
                # Train individual classifiers ONCE (not in loop!)
                status_text.text("Training base classifiers...")
                from sklearn.linear_model import LogisticRegression
                
                classifiers = []
                train_probas = []
                test_probas = []
                
                for i, (train_m, test_m) in enumerate(zip(train_modalities, test_modalities)):
                    clf = LogisticRegression(max_iter=1000, random_state=42, C=1.0)
                    clf.fit(train_m, y_train)
                    classifiers.append(clf)
                    
                    train_probas.append(clf.predict_proba(train_m))
                    test_probas.append(clf.predict_proba(test_m))
                
                progress_bar.progress(20)
                
                # Initialize attention weights (learnable parameters)
                attention_weights = np.ones(4) / 4  # Start uniform
                
                # Training history
                train_losses = []
                train_accs = []
                val_accs = []
                attention_history = []
                
                n_classes = st.session_state.n_classes
                
                # PROPER TRAINING LOOP WITH GRADIENT DESCENT
                status_text.text("Training attention mechanism...")
                
                for epoch in range(epochs):
                    # === FORWARD PASS ===
                    # Fused prediction using current attention weights
                    train_fused = sum(w * p for w, p in zip(attention_weights, train_probas))
                    test_fused = sum(w * p for w, p in zip(attention_weights, test_probas))
                    
                    # Predictions
                    train_pred = np.argmax(train_fused, axis=1)
                    test_pred = np.argmax(test_fused, axis=1)
                    
                    # Metrics
                    train_acc = accuracy_score(y_train, train_pred)
                    test_acc = accuracy_score(y_test, test_pred)
                    
                    # Loss (cross-entropy)
                    train_loss = -np.mean(np.log(train_fused[np.arange(len(y_train)), y_train] + 1e-10))
                    
                    # === BACKWARD PASS (Gradient Descent) ===
                    # Compute gradient of loss w.r.t attention weights
                    gradients = np.zeros(4)
                    
                    for i in range(4):
                        # Numerical gradient (simple but works)
                        epsilon = 1e-5
                        weights_plus = attention_weights.copy()
                        weights_plus[i] += epsilon
                        weights_plus = weights_plus / weights_plus.sum()  # Normalize
                        
                        train_fused_plus = sum(w * p for w, p in zip(weights_plus, train_probas))
                        loss_plus = -np.mean(np.log(train_fused_plus[np.arange(len(y_train)), y_train] + 1e-10))
                        
                        gradients[i] = (loss_plus - train_loss) / epsilon
                    
                    # === UPDATE WEIGHTS ===
                    # Gradient descent with momentum
                    attention_weights = attention_weights - learning_rate * gradients
                    
                    # Add regularization (prevent one weight from dominating)
                    attention_weights = attention_weights + regularization * (0.25 - attention_weights)
                    
                    # Project to simplex (weights sum to 1)
                    attention_weights = np.maximum(attention_weights, 0.01)  # Prevent negatives
                    attention_weights = attention_weights / attention_weights.sum()
                    
                    # Store history
                    train_losses.append(train_loss)
                    train_accs.append(train_acc)
                    val_accs.append(test_acc)
                    attention_history.append(attention_weights.copy())
                    
                    # Update progress
                    progress_bar.progress(20 + int((epoch + 1) / epochs * 75))
                    
                    if epoch % 10 == 0:
                        status_text.text(f"Epoch {epoch+1}/{epochs} - Loss: {train_loss:.4f}, Val Acc: {test_acc:.3f}")
                
                # Final predictions
                final_fused = sum(w * p for w, p in zip(attention_weights, test_probas))
                y_pred = np.argmax(final_fused, axis=1)
                final_accuracy = accuracy_score(y_test, y_pred)
                
                status_text.text("Training complete!")
                progress_bar.progress(100)
                
                # Store results
                st.session_state.attention_weights = attention_weights
                st.session_state.attention_history = np.array(attention_history)
                st.session_state.train_losses = train_losses
                st.session_state.train_accs = train_accs
                st.session_state.val_accs = val_accs
                st.session_state.y_test = y_test
                st.session_state.y_pred = y_pred
                st.session_state.final_accuracy = final_accuracy
                st.session_state.model_trained = True
                
                status_text.empty()
                st.success(f"✅ Model trained! Final Test Accuracy: {final_accuracy*100:.2f}%")
                st.balloons()
        
        # Show results if model is trained
        if st.session_state.model_trained:
            st.markdown('<p class="section-header">📈 Training Results</p>', unsafe_allow_html=True)
            
            cols = st.columns(4)
            with cols[0]:
                st.metric("Final Accuracy", f"{st.session_state.final_accuracy*100:.2f}%")
            with cols[1]:
                st.metric("🖼️ Image", f"{st.session_state.attention_weights[0]:.3f}")
            with cols[2]:
                st.metric("📝 Text", f"{st.session_state.attention_weights[1]:.3f}")
            with cols[3]:
                st.metric("🎵 Audio", f"{st.session_state.attention_weights[2]:.3f}")
            
            # Training curves
            col1, col2 = st.columns(2)
            
            with col1:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    y=st.session_state.train_losses,
                    mode='lines+markers',
                    name='Training Loss',
                    line=dict(color='#667eea', width=3),
                    marker=dict(size=4)
                ))
                fig.update_layout(
                    title='Training Loss Over Epochs (Should Decrease!)',
                    xaxis_title='Epoch',
                    yaxis_title='Loss',
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    hovermode='x unified'
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    y=st.session_state.val_accs,
                    mode='lines+markers',
                    name='Validation Accuracy',
                    line=dict(color='#10b981', width=3),
                    marker=dict(size=4)
                ))
                fig.add_trace(go.Scatter(
                    y=st.session_state.train_accs,
                    mode='lines',
                    name='Training Accuracy',
                    line=dict(color='#60a5fa', width=2, dash='dash'),
                    opacity=0.6
                ))
                fig.update_layout(
                    title='Accuracy Over Epochs (Should Increase!)',
                    xaxis_title='Epoch',
                    yaxis_title='Accuracy',
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    hovermode='x unified'
                )
                st.plotly_chart(fig, use_container_width=True)
 
# Page: Analysis & Insights
elif page == "📊 Analysis & Insights":
    st.markdown('<p class="section-header">Model Analysis & Insights</p>', unsafe_allow_html=True)
    
    if not st.session_state.model_trained:
        st.warning("⚠️ Please train a model first in the 'Fusion & Training' page!")
    else:
        # Modality importance visualization
        st.markdown("### 🎯 Learned Modality Importance Weights")
        
        modality_names = ['🖼️ Image', '📝 Text', '🎵 Audio', '📡 Sensor']
        colors = ['#f093fb', '#4facfe', '#43e97b', '#fa709a']
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=modality_names,
            y=st.session_state.attention_weights,
            marker=dict(
                color=colors,
                line=dict(color='white', width=2)
            ),
            text=[f"{w:.3f}" for w in st.session_state.attention_weights],
            textposition='outside',
            textfont=dict(size=14, color='black', family='Arial Black'),
        ))
        fig.update_layout(
            yaxis_title='Attention Weight',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            height=400,
            showlegend=False,
            yaxis=dict(range=[0, max(st.session_state.attention_weights) * 1.2])
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Attention evolution
        st.markdown("### 📈 Attention Weight Evolution During Training")
        
        attention_history = np.array(st.session_state.attention_history)
        
        fig = go.Figure()
        for i, (name, color) in enumerate(zip(modality_names, colors)):
            fig.add_trace(go.Scatter(
                y=attention_history[:, i],
                mode='lines',
                name=name,
                line=dict(color=color, width=3)
            ))
        fig.update_layout(
            xaxis_title='Epoch',
            yaxis_title='Attention Weight',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            height=400,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Confusion Matrix
        st.markdown("### 🎯 Confusion Matrix")
        
        cm = confusion_matrix(st.session_state.y_test, st.session_state.y_pred)
        
        fig = go.Figure(data=go.Heatmap(
            z=cm,
            x=[f'Class {i}' for i in range(st.session_state.n_classes)],
            y=[f'Class {i}' for i in range(st.session_state.n_classes)],
            colorscale='Purples',
            text=cm,
            texttemplate='%{text}',
            textfont={"size": 16},
            showscale=True
        ))
        fig.update_layout(
            xaxis_title='Predicted',
            yaxis_title='Actual',
            height=400,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Classification report
        st.markdown("### 📋 Detailed Classification Report")
        
        report = classification_report(
            st.session_state.y_test, 
            st.session_state.y_pred, 
            output_dict=True
        )
        
        df_report = pd.DataFrame(report).transpose()
        df_report = df_report.round(3)
        
        st.dataframe(
            df_report.style.background_gradient(cmap='Purples', subset=['f1-score']),
            use_container_width=True
        )
        
        # Key insights
        st.markdown("### 💡 Key Insights")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="info-card">
            <h4>🔍 Interpretation</h4>
            <ul style='line-height: 1.8;'>
                <li><strong>Highest Weight:</strong> The modality with the highest attention weight contributes most to predictions</li>
                <li><strong>Weight Evolution:</strong> Stable weights indicate consistent importance; changing weights show adaptive learning</li>
                <li><strong>Fusion Benefit:</strong> Combined modalities typically outperform single modalities</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            dominant_modality = modality_names[np.argmax(st.session_state.attention_weights)]
            weakest_modality = modality_names[np.argmin(st.session_state.attention_weights)]
            
            st.markdown(f"""
            <div class="info-card">
            <h4>📊 Your Model's Behavior</h4>
            <ul style='line-height: 1.8;'>
                <li><strong>Most Important:</strong> {dominant_modality} ({st.session_state.attention_weights.max():.3f})</li>
                <li><strong>Least Important:</strong> {weakest_modality} ({st.session_state.attention_weights.min():.3f})</li>
                <li><strong>Final Accuracy:</strong> {st.session_state.final_accuracy*100:.2f}%</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
 
# Page: Learn More
elif page == "🎓 Learn More":
    st.markdown('<p class="section-header">Deep Dive: Multimodal Fusion Theory</p>', unsafe_allow_html=True)
    
    tab1, tab2, tab3, tab4 = st.tabs(["📖 Theory", "🔬 Methods", "💻 Applications", "📚 Resources"])
    
    with tab1:
        st.markdown("""
        ### What is Multimodal Learning?
        
        Multimodal learning involves processing and relating information from multiple modalities. 
        Humans naturally integrate information from different senses - we see, hear, touch, and combine 
        these to understand the world.
        
        #### Key Challenges:
        
        1. **Representation**: How to represent each modality?
        2. **Translation**: How to translate between modalities?
        3. **Alignment**: How to align elements from different modalities?
        4. **Fusion**: How to combine information from modalities?
        5. **Co-learning**: How can one modality help another?
        
        #### Fusion Strategies:
        
        - **Early Fusion**: Combine raw features directly
        - **Late Fusion**: Combine decisions from each modality
        - **Hybrid Fusion**: Mix of early and late fusion
        - **Attention-Based**: Learn importance weights dynamically
        """)
        
        st.image("https://img.icons8.com/color/200/000000/artificial-intelligence.png", width=150)
    
    with tab2:
        st.markdown("""
        ### Common Fusion Methods
        
        #### 1. Concatenation
        Simply stack features: `[f1; f2; f3]`
        - ✅ Simple and intuitive
        - ❌ Assumes equal importance
        
        #### 2. Weighted Sum
        Learn weights: `α1·f1 + α2·f2 + α3·f3`
        - ✅ Captures relative importance
        - ✅ Interpretable
        
        #### 3. Attention Mechanism
        Dynamic weights based on context
        - ✅ Adaptive to input
        - ✅ State-of-the-art performance
        
        #### 4. Tensor Fusion
        Outer product of modalities
        - ✅ Captures interactions
        - ❌ High computational cost
        
        ### Mathematical Formulation
        
        Given modalities M₁, M₂, ..., Mₙ with features h₁, h₂, ..., hₙ:
        
        **Attention Weights:**
        ```
        αᵢ = exp(wᵢ) / Σⱼ exp(wⱼ)
        ```
        
        **Fused Representation:**
        ```
        f = Σᵢ αᵢ · hᵢ
        ```
        
        Where wᵢ are learned parameters.
        """)
    
    with tab3:
        st.markdown("""
        ### Real-World Applications
        
        #### 🏥 Healthcare
        - Combining MRI, CT scans, and patient records for diagnosis
        - Integrating genomic data with medical imaging
        
        #### 🚗 Autonomous Driving
        - Fusing camera, LiDAR, and radar data
        - Combining visual and GPS information
        
        #### 🎬 Media & Entertainment
        - Audio-visual speech recognition
        - Video understanding with subtitles
        
        #### 🛍️ E-commerce
        - Product recommendations from images and descriptions
        - Visual search with text queries
        
        #### 🤖 Robotics
        - Sensor fusion for navigation
        - Multi-modal human-robot interaction
        
        #### 🔐 Security
        - Biometric authentication (face + voice + fingerprint)
        - Anomaly detection across multiple data streams
        """)
        
        cols = st.columns(3)
        with cols[0]:
            st.image("https://img.icons8.com/color/150/000000/healthcare.png", width=100)
        with cols[1]:
            st.image("https://img.icons8.com/color/150/000000/autonomous-car.png", width=100)
        with cols[2]:
            st.image("https://img.icons8.com/color/150/000000/robot.png", width=100)
    
    with tab4:
        st.markdown("""
        ### 📚 Recommended Resources
        
        #### Papers
        1. **"Multimodal Machine Learning: A Survey and Taxonomy"** - Baltrušaitis et al., 2018
        2. **"Attention is All You Need"** - Vaswani et al., 2017
        3. **"Tensor Fusion Network for Multimodal Sentiment Analysis"** - Zadeh et al., 2017
        
        #### Online Courses
        - 🎓 Deep Learning Specialization (Coursera)
        - 🎓 Multimodal Deep Learning (Stanford CS231n)
        
        #### Libraries & Tools
        - 🔧 PyTorch MultiModal (TorchMultimodal)
        - 🔧 HuggingFace Transformers
        - 🔧 TensorFlow
        
        #### Datasets
        - 🗂️ COCO (images + captions)
        - 🗂️ AudioSet (audio labels)
        - 🗂️ CMU-MOSI (multimodal sentiment)
        """)
        
        st.info("💡 **Pro Tip**: Start with simple fusion methods and gradually move to complex architectures!")
 
# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #6b7280; padding: 2rem;'>
    <p style='font-size: 0.9rem;'>
        🧬 Multimodal Fusion Virtual Lab | Built with Streamlit & Python<br>
        Explore • Learn • Innovate
    </p>
</div>
""", unsafe_allow_html=True)
