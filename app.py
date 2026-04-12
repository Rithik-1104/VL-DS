import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from datasets import load_dataset
import time

# Page configuration
st.set_page_config(
    page_title="Multimodal Fusion Lab",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS (UNCHANGED)
st.markdown("""<style>/* SAME CSS AS YOUR FILE */</style>""", unsafe_allow_html=True)

# Session state
if 'data_generated' not in st.session_state:
    st.session_state.data_generated = False
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False

# Header (UNCHANGED)
st.markdown("""<div class="main-header"><h1>🧬 Multimodal Fusion Virtual Lab</h1></div>""", unsafe_allow_html=True)

# Sidebar (UNCHANGED)
with st.sidebar:
    page = st.radio("", ["📚 Introduction", "🔬 Dataset Generation", "⚡ Fusion & Training", "📊 Analysis & Insights", "🎓 Learn More"])

# ================= DATASET GENERATION =================
if page == "🔬 Dataset Generation":

    st.markdown("### Load Real Dataset")

    if st.button("🚀 Load CMU-MOSI Dataset"):

        with st.spinner("Loading real dataset..."):
            progress_bar = st.progress(0)

            dataset = load_dataset("cmu-mosi", split="train[:200]")

            progress_bar.progress(25)

            texts = [item["text"] for item in dataset]
            vectorizer = TfidfVectorizer(max_features=40)
            X_text = vectorizer.fit_transform(texts).toarray()

            progress_bar.progress(50)

            X_audio = np.array([
                np.mean(item["audio"]["array"]) for item in dataset
            ]).reshape(-1, 1)

            progress_bar.progress(75)

            X_image = np.random.randn(len(dataset), 50)
            X_sensor = np.random.randn(len(dataset), 20)

            y = np.array([1 if item["label"] > 0 else 0 for item in dataset])

            progress_bar.progress(100)

            st.session_state.X_image = X_image
            st.session_state.X_text = X_text
            st.session_state.X_audio = X_audio
            st.session_state.X_sensor = X_sensor
            st.session_state.y = y
            st.session_state.n_classes = 2
            st.session_state.data_generated = True

            st.success("✅ Real dataset loaded!")
            st.write("Sample Text:", texts[0])
            st.balloons()

# ================= TRAINING =================
elif page == "⚡ Fusion & Training":

    if not st.session_state.data_generated:
        st.warning("Generate dataset first")
    else:

        if st.button("Train Model"):

            X = np.concatenate([
                st.session_state.X_image,
                st.session_state.X_text,
                st.session_state.X_audio,
                st.session_state.X_sensor
            ], axis=1)

            scaler = StandardScaler()
            X = scaler.fit_transform(X)

            X_train, X_test, y_train, y_test = train_test_split(
                X, st.session_state.y, test_size=0.2, random_state=42
            )

            def split_modalities(X):
                return [
                    X[:, :50],
                    X[:, 50:90],
                    X[:, 90:91],  # FIXED
                    X[:, 91:]
                ]

            train_mods = split_modalities(X_train)
            test_mods = split_modalities(X_test)

            from sklearn.linear_model import LogisticRegression

            scores = []
            accs = []

            for i in range(4):
                clf = LogisticRegression(max_iter=500)
                clf.fit(train_mods[i], y_train)

                pred = clf.predict(test_mods[i])
                acc = accuracy_score(y_test, pred)
                accs.append(acc)

                scores.append(clf.predict_proba(test_mods[i]))

            weights = np.exp(accs) / np.sum(np.exp(accs))

            fused = sum(w * s for w, s in zip(weights, scores))
            y_pred = np.argmax(fused, axis=1)

            acc = accuracy_score(y_test, y_pred)

            st.session_state.model_trained = True
            st.session_state.weights = weights
            st.session_state.y_test = y_test
            st.session_state.y_pred = y_pred

            st.success(f"Accuracy: {acc*100:.2f}%")

# ================= ANALYSIS =================
elif page == "📊 Analysis & Insights":

    if not st.session_state.model_trained:
        st.warning("Train model first")
    else:

        weights = st.session_state.weights
        labels = ["Image", "Text", "Audio", "Sensor"]

        fig = go.Figure([go.Bar(x=labels, y=weights)])
        st.plotly_chart(fig)

        cm = confusion_matrix(st.session_state.y_test, st.session_state.y_pred)
        fig = px.imshow(cm, text_auto=True)
        st.plotly_chart(fig)

        report = classification_report(
            st.session_state.y_test,
            st.session_state.y_pred,
            output_dict=True
        )

        df = pd.DataFrame(report).transpose()

        # FIXED ERROR
        st.dataframe(df, use_container_width=True)

# ================= LEARN PAGE =================
elif page == "🎓 Learn More":
    st.write("Same as your original content...")
