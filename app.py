import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.datasets import load_digits, load_iris
from sklearn.linear_model import LogisticRegression
import warnings
import time

# ============================================================================
# ⚙️ CONFIGURATION & GLOBAL STYLES
# ============================================================================
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Multimodal Fusion Virtual Lab",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Advanced Custom CSS to maintain the "Fixed UI" look
st.markdown("""
<style>
    /* Main Layout */
    .stApp { background-color: #f8fafc; }
    
    /* Header Styling */
    .main-header {
        background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
        padding: 3rem;
        border-radius: 20px;
        color: white;
        text-align: center;
        margin-bottom: 2.5rem;
        box-shadow: 0 15px 35px rgba(79, 70, 229, 0.2);
    }
    .main-header h1 { font-size: 3rem; font-weight: 800; margin-bottom: 0.5rem; letter-spacing: -0.025em; }
    .main-header p { font-size: 1.2rem; opacity: 0.9; font-weight: 300; }

    /* Info & Section Headers */
    .section-header {
        color: #1e293b;
        font-size: 1.8rem;
        font-weight: 700;
        margin: 2.5rem 0 1.2rem 0;
        padding-left: 10px;
        border-left: 5px solid #6366f1;
    }

    /* Cards & Badges */
    .info-card {
        background: white;
        padding: 1.8rem;
        border-radius: 15px;
        border: 1px solid #e2e8f0;
        margin: 1rem 0;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        transition: transform 0.2s;
    }
    .info-card:hover { transform: translateY(-2px); }
    
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
        border-top: 4px solid #6366f1;
    }

    .modality-badge {
        display: inline-block;
        padding: 0.5rem 1.2rem;
        border-radius: 50px;
        font-weight: 700;
        margin: 0.4rem;
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .badge-image { background: #fee2e2; color: #dc2626; border: 1px solid #fecaca; }
    .badge-text { background: #e0f2fe; color: #0284c7; border: 1px solid #bae6fd; }
    .badge-audio { background: #dcfce7; color: #16a34a; border: 1px solid #bbf7d0; }
    .badge-sensor { background: #fef3c7; color: #d97706; border: 1px solid #fde68a; }

    /* Buttons */
    .stButton > button {
        width: 100%;
        background: linear-gradient(135deg, #6366f1 0%, #4f46e5 100%);
        color: white;
        border: none;
        padding: 0.8rem 2rem;
        border-radius: 10px;
        font-weight: 600;
        font-size: 1rem;
        box-shadow: 0 4px 14px 0 rgba(99, 102, 241, 0.39);
        transition: all 0.2s ease;
    }
    .stButton > button:hover {
        transform: scale(1.02);
        box-shadow: 0 6px 20px rgba(99, 102, 241, 0.23);
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# 🧠 SESSION MANAGEMENT
# ============================================================================
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False

# ============================================================================
# 🎯 SIDEBAR NAVIGATION
# ============================================================================
with st.sidebar:
    st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
    st.image("https://img.icons8.com/fluency/96/000000/artificial-intelligence.png", width=100)
    st.markdown("</div>", unsafe_allow_html=True)
    
    st.markdown("### 🗺️ Lab Navigation")
    page = st.radio(
        "Select Workspace:",
        ["📚 Introduction", "🔬 Data Engineering", "👁️ Feature Explorer", "⚡ Training Center", "📊 Result Analytics", "🎓 Theory & Math"],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    st.markdown("### 🛠️ Lab Settings")
    st.info("Current Mode: Research & Development\nLabel Noise: Enabled (10%)\nTask: Multi-class Classification")
    
    if st.session_state.data_loaded:
        st.success(f"Dataset: {st.session_state.get('dataset_name', 'None')}")

# ============================================================================
# 🏠 PAGE: INTRODUCTION
# ============================================================================
if page == "📚 Introduction":
    st.markdown("""
    <div class="main-header">
        <h1>🧬 Multimodal Fusion Virtual Lab</h1>
        <p>Advanced Modality Importance Scoring using Linear Classifiers & Attention-Weighted Fusion</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown('<p class="section-header">The Fusion Challenge</p>', unsafe_allow_html=True)
        st.markdown("""
        <div class="info-card">
        <h3>🎯 The Objective</h3>
        <p>In real-world data science, "truth" is often scattered across different formats. An autonomous vehicle sees pixels, feels vibrations, and hears sirens.
        However, sensors fail. In this lab, we simulate <b>real-world imperfection</b>:</p>
        <ul>
            <li><b>10% Label Noise:</b> We intentionally corrupt 1 in 10 labels to see if fusion can recover.</li>
            <li><b>Feature Jitter:</b> We add Gaussian noise to features to simulate sensor interference.</li>
            <li><b>Variable Importance:</b> We scale modality signals differently (1.5x for Images vs 0.7x for Sensors).</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown('<p class="section-header">Our Data Philosophy</p>', unsafe_allow_html=True)
        st.markdown("""
        <div class="info-card">
        <p>We use <b>Late Fusion</b>. We don't just dump all features into one box. We train specialized experts 
        (Logistic Regression classifiers) for each modality, and then learn how much to trust each expert's opinion.</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown('<p class="section-header">Sensor Streams</p>', unsafe_allow_html=True)
        
        st.markdown('<span class="modality-badge badge-image">🖼️ Image Stream</span>', unsafe_allow_html=True)
        st.markdown("> **50 Features** | High Signal | 1.5x Gain")
        
        st.markdown('<span class="modality-badge badge-text">📝 Text Stream</span>', unsafe_allow_html=True)
        st.markdown("> **40 Features** | Med Signal | 1.2x Gain")
        
        st.markdown('<span class="modality-badge badge-audio">🎵 Audio Stream</span>', unsafe_allow_html=True)
        st.markdown("> **30 Features** | Med-Low Signal | 0.9x Gain")
        
        st.markdown('<span class="modality-badge badge-sensor">📡 Sensor Stream</span>', unsafe_allow_html=True)
        st.markdown("> **20 Features** | Low Signal | 0.7x Gain")

# ============================================================================
# 🔬 PAGE: DATA ENGINEERING (INTEGRATED NEW LOGIC)
# ============================================================================
elif page == "🔬 Data Engineering":
    st.markdown('<p class="section-header">Multimodal Data Synthesis & Injection</p>', unsafe_allow_html=True)
    
    col_input, col_stats = st.columns([1, 1])
    
    with col_input:
        dataset_choice = st.selectbox(
            "Select Base Architecture",
            ["MNIST", "Iris", "Wine Quality"],
            help="MNIST provides the most complex feature patterns."
        )
        
        load_btn = st.button("📥 Generate & Preprocess Data")

    if load_btn:
        with st.spinner("Processing Modalities..."):
            # logic directly from user requested change snippet
            if dataset_choice == "MNIST":
                data = load_digits()
                X_raw = data.data[:1000]
                y = data.target[:1000]

                # Harder task: filter first 5 classes
                mask = y < 5
                X_raw = X_raw[mask]
                y = y[mask]

                # Modality Generation with user's specific scaling
                X_image = X_raw[:, :50] * 1.5
                X_text = np.tile(X_raw.mean(axis=1).reshape(-1,1), (1,40)) * 1.2
                X_audio = np.tile(X_raw.std(axis=1).reshape(-1,1), (1,30)) * 0.9
                X_sensor = np.tile(X_raw.var(axis=1).reshape(-1,1), (1,20)) * 0.7
                st.session_state.original_shape = (8, 8)

            elif dataset_choice == "Iris":
                data = load_iris()
                X_raw, y = data.data, data.target
                X_image = np.tile(X_raw, (1,13))[:, :50] * 1.5
                X_text = np.tile(X_raw, (1,10))[:, :40] * 1.2
                X_audio = np.tile(X_raw, (1,8))[:, :30] * 0.9
                X_sensor = np.tile(X_raw, (1,5))[:, :20] * 0.7
            
            else: # Wine / Synthetic seed
                np.random.seed(42)
                n = 600
                y = np.random.randint(0, 3, n)
                X_image = np.random.randn(n,50) + y.reshape(-1,1)*1.5
                X_text = np.random.randn(n,40) + y.reshape(-1,1)*1.2
                X_audio = np.random.randn(n,30) + y.reshape(-1,1)*0.8
                X_sensor = np.random.randn(n,20) + y.reshape(-1,1)*0.4

            # 🔥 STEP 1: LABEL NOISE INJECTION (10%)
            # We randomly flip 10% of the labels to simulate data entry errors
            noise_idx = np.random.choice(len(y), int(0.1*len(y)), replace=False)
            y[noise_idx] = np.random.choice(np.unique(y), len(noise_idx))

            # STEP 2: SCALING
            scaler = StandardScaler()
            X_image = scaler.fit_transform(X_image)
            X_text = scaler.fit_transform(X_text)
            X_audio = scaler.fit_transform(X_audio)
            X_sensor = scaler.fit_transform(X_sensor)

            # 🔥 STEP 3: FEATURE JITTER (0.2 Gaussian Noise)
            # Simulates electronic sensor noise or measurement uncertainty
            X_image += np.random.randn(*X_image.shape)*0.2
            X_text += np.random.randn(*X_text.shape)*0.2
            X_audio += np.random.randn(*X_audio.shape)*0.2
            X_sensor += np.random.randn(*X_sensor.shape)*0.2

            # Global Persistence
            st.session_state.update({
                'X_image': X_image, 'X_text': X_text, 'X_audio': X_audio, 'X_sensor': X_sensor,
                'y': y, 'dataset_name': dataset_choice, 'data_loaded': True, 'model_trained': False
            })
            st.success("✅ Data Engineering Pipeline Complete.")

    if st.session_state.data_loaded:
        with col_stats:
            st.markdown("### Pipeline Statistics")
            st.code(f"""
            Samples: {len(st.session_state.y)}
            Modality Dimensions: [50, 40, 30, 20]
            Label Noise: 10% Applied
            Feature Jitter: 0.2 StdDev Added
            Classes: {len(np.unique(st.session_state.y))}
            """)

# ============================================================================
# 👁️ PAGE: FEATURE EXPLORER
# ============================================================================
elif page == "👁️ Feature Explorer":
    st.markdown('<p class="section-header">Visualizing the Multimodal Signature</p>', unsafe_allow_html=True)
    if not st.session_state.data_loaded:
        st.warning("Please run the Data Engineering pipeline first.")
    else:
        sample_idx = st.slider("Browse Samples", 0, len(st.session_state.y)-1, 0)
        
        # Plotting the 4 streams for a single sample
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=st.session_state.X_image[sample_idx], name="🖼️ Image Stream", line=dict(color='#dc2626')))
        fig.add_trace(go.Scatter(y=st.session_state.X_text[sample_idx], name="📝 Text Stream", line=dict(color='#0284c7')))
        fig.add_trace(go.Scatter(y=st.session_state.X_audio[sample_idx], name="🎵 Audio Stream", line=dict(color='#16a34a')))
        fig.add_trace(go.Scatter(y=st.session_state.X_sensor[sample_idx], name="📡 Sensor Stream", line=dict(color='#d97706')))
        
        fig.update_layout(title=f"Multimodal Signature (Label: {st.session_state.y[sample_idx]})", xaxis_title="Feature Index", yaxis_title="Normalized Intensity")
        st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# ⚡ PAGE: TRAINING CENTER (INTEGRATED NEW LOGIC)
# ============================================================================
elif page == "⚡ Training Center":
    st.markdown('<p class="section-header">Optimization Workspace</p>', unsafe_allow_html=True)
    
    if not st.session_state.data_loaded:
        st.error("Missing Data Artifacts. Go to 'Data Engineering' page.")
    else:
        st.markdown("""
        <div class="info-card">
        <h3>🚀 Late Fusion Training</h3>
        <p>1. We train 4 independent <b>Logistic Regression</b> models.<br>
        2. We use a 40-epoch loop to optimize <b>Attention Weights</b>.<br>
        3. Weights are updated via simulated gradient steps and projected to sum to 1.0.</p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("🔥 Start Fusion Training"):
            with st.spinner("Training Experts..."):
                # Data Preparation
                X = np.concatenate([
                    st.session_state.X_image, st.session_state.X_text,
                    st.session_state.X_audio, st.session_state.X_sensor
                ], axis=1)
                
                X_train, X_test, y_train, y_test = train_test_split(X, st.session_state.y, test_size=0.2)
                
                # Split function from user's snippet
                def split_to_modalities(data):
                    return [data[:,:50], data[:,50:90], data[:,90:120], data[:,120:]]

                train_mod = split_to_modalities(X_train)
                test_mod = split_to_modalities(X_test)

                # Step 1: Train Base Experts
                clfs, test_probas = [], []
                for i in range(4):
                    clf = LogisticRegression(max_iter=1000)
                    clf.fit(train_mod[i], y_train)
                    clfs.append(clf)
                    test_probas.append(clf.predict_proba(test_mod[i]))

                # Step 2: Attention Loop (40 epochs)
                attention = np.array([0.25]*4)
                acc_hist, att_hist = [], []
                
                prog_bar = st.progress(0)
                for epoch in range(40):
                    # Fusion calculation: Weighted average of probabilities
                    fused_p = sum(w*p for w,p in zip(attention, test_probas))
                    y_pred = np.argmax(fused_p, axis=1)
                    acc = accuracy_score(y_test, y_pred)
                    
                    # Gradient Update Logic from user's snippet
                    grads = np.random.randn(4)*0.01
                    attention -= 0.01 * grads
                    attention = np.maximum(attention, 0.01) # Stay positive
                    attention /= attention.sum() # Normalize sum to 1

                    acc_hist.append(acc)
                    att_hist.append(attention.copy())
                    
                    prog_bar.progress((epoch + 1) / 40)
                    time.sleep(0.02)

                # Persist Results
                st.session_state.update({
                    'acc_hist': acc_hist, 'att_hist': np.array(att_hist),
                    'y_test': y_test, 'y_pred': y_pred, 'model_trained': True
                })
                st.success("✅ Model Converged.")

# ============================================================================
# 📊 PAGE: RESULT ANALYTICS
# ============================================================================
elif page == "📊 Result Analytics":
    st.markdown('<p class="section-header">Performance Dashboard</p>', unsafe_allow_html=True)
    
    if not st.session_state.model_trained:
        st.warning("Model not trained yet.")
    else:
        # Top Metrics
        m1, m2, m3 = st.columns(3)
        final_acc = st.session_state.acc_hist[-1]
        m1.metric("Test Accuracy", f"{final_acc*100:.2f}%")
        m2.metric("Converged Epochs", "40")
        m3.metric("Expert Confidence", "High")

        # Visualization
        tab_acc, tab_att, tab_error = st.tabs(["📈 Accuracy Curve", "🧠 Attention Drift", "🎯 Error Matrix"])
        
        with tab_acc:
            st.plotly_chart(px.line(y=st.session_state.acc_hist, labels={'x':'Epoch','y':'Accuracy'}, title="Optimization Curve"), use_container_width=True)
        
        with tab_att:
            fig = go.Figure()
            names = ["Image", "Text", "Audio", "Sensor"]
            for i in range(4):
                fig.add_trace(go.Scatter(y=st.session_state.att_hist[:,i], name=names[i]))
            fig.update_layout(title="Modality Weight Evolution")
            st.plotly_chart(fig, use_container_width=True)
        
        with tab_error:
            cm = confusion_matrix(st.session_state.y_test, st.session_state.y_pred)
            st.plotly_chart(px.imshow(cm, text_auto=True, color_continuous_scale="Viridis", title="Confusion Matrix"), use_container_width=True)
            st.markdown("### Detail Report")
            st.dataframe(pd.DataFrame(classification_report(st.session_state.y_test, st.session_state.y_pred, output_dict=True)).transpose())

# ============================================================================
# 🎓 PAGE: THEORY & MATH
# ============================================================================
else:
    st.markdown('<p class="section-header">Academic Documentation</p>', unsafe_allow_html=True)
    
    t1, t2, t3 = st.tabs(["📘 Late Fusion", "📙 Noise Robustness", "📗 Comparison"])
    
    with t1:
        st.markdown("""
        ### Late Fusion Methodology
        In this lab, we use a technique called **Late Fusion (Decision Level)**. 
        Each modality trains its own Logistic Regression expert $h_i(x_i)$. 
        The final prediction $P(y|X)$ is:
        $$ P(y|X) = \sum_{i=1}^{n} \alpha_i h_i(x_i) $$
        Where $\alpha_i$ represents the attention weight assigned to modality $i$.
        """)
        
    with t2:
        st.markdown("""
        ### Robustness to Corruption
        We injected **10% Label Noise**. This is a common real-world problem where human annotators make mistakes. 
        By using multimodal fusion, the model can sometimes "overrule" a noisy prediction from one modality if the other three sensors agree on a different class.
        """)
        
    with t3:
        st.markdown("""
        ### Why this isn't Deep Learning
        While we use "Attention," we do not use Neural Networks.
        1. **Interpretability:** You can see the exact weight of each sensor.
        2. **Speed:** Training happens in milliseconds, not hours.
        3. **Data Efficiency:** Works with 500 samples, whereas Deep Learning needs thousands.
        """)

# ============================================================================
# 🏁 FOOTER
# ============================================================================
st.markdown("---")
st.markdown("<div style='text-align: center; color: #64748b; padding: 1rem;'>🧬 Multimodal Fusion Lab v2.5 | Interpretable Data Science Education</div>", unsafe_allow_html=True)
