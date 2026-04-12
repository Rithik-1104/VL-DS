import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.linear_model import SGDClassifier

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Multimodal Fusion Lab", layout="wide")

st.title("🧬 Multimodal Fusion Virtual Lab")

# ---------------- SESSION ----------------
if "data_generated" not in st.session_state:
    st.session_state.data_generated = False
if "model_trained" not in st.session_state:
    st.session_state.model_trained = False

# ---------------- SIDEBAR ----------------
page = st.sidebar.radio(
    "Navigation",
    ["Dataset Generation", "Fusion & Training", "Analysis"]
)

# ---------------- DATASET ----------------
if page == "Dataset Generation":

    st.header("Generate Multimodal Dataset")

    n_samples = st.slider("Samples", 100, 2000, 500)
    noise_level = st.slider("Noise", 0.0, 0.5, 0.1)

    if st.button("Generate"):

        def generate(n, f, imp):
            X = np.random.randn(n, f)
            X += imp * np.random.randn(n, f)
            return X

        X_image = generate(n_samples, 50, 2.5)
        X_text  = generate(n_samples, 40, 1.8)
        X_audio = generate(n_samples, 30, 1.2)
        X_sensor= generate(n_samples, 20, 0.8)

        # noise
        X_image += noise_level * np.random.randn(*X_image.shape)
        X_text  += noise_level * np.random.randn(*X_text.shape)
        X_audio += noise_level * np.random.randn(*X_audio.shape)
        X_sensor+= noise_level * np.random.randn(*X_sensor.shape)

        # real relationship
        y_signal = (
            0.5 * X_image[:,0] +
            0.3 * X_text[:,0] +
            0.1 * X_audio[:,0] +
            0.1 * X_sensor[:,0]
        )

        y = (y_signal > np.median(y_signal)).astype(int)

        st.session_state.X_image = X_image
        st.session_state.X_text = X_text
        st.session_state.X_audio = X_audio
        st.session_state.X_sensor = X_sensor
        st.session_state.y = y
        st.session_state.data_generated = True

        st.success("Dataset Generated")

# ---------------- TRAINING ----------------
elif page == "Fusion & Training":

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
                X, st.session_state.y, test_size=0.2
            )

            def split(X):
                return [
                    X[:,:50],
                    X[:,50:90],
                    X[:,90:120],
                    X[:,120:]
                ]

            train_mod = split(X_train)
            test_mod = split(X_test)

            models = [
                SGDClassifier(loss="log_loss"),
                SGDClassifier(loss="log_loss"),
                SGDClassifier(loss="log_loss"),
                SGDClassifier(loss="log_loss")
            ]

            epochs = 30
            acc_history = []
            att_history = []

            for epoch in range(epochs):

                modality_accs = []
                scores = []

                for i in range(4):
                    models[i].partial_fit(train_mod[i], y_train, classes=np.unique(y_train))

                    pred = models[i].predict(test_mod[i])
                    acc = accuracy_score(y_test, pred)
                    modality_accs.append(acc)

                    scores.append(models[i].predict_proba(test_mod[i]))

                weights = np.exp(modality_accs) / np.sum(np.exp(modality_accs))
                att_history.append(weights)

                fused = sum(w*s for w,s in zip(weights, scores))
                y_pred = np.argmax(fused, axis=1)

                acc = accuracy_score(y_test, y_pred)
                acc_history.append(acc)

            st.session_state.acc_history = acc_history
            st.session_state.att_history = att_history
            st.session_state.y_test = y_test
            st.session_state.y_pred = y_pred
            st.session_state.model_trained = True

            st.success("Training Done")

# ---------------- ANALYSIS ----------------
elif page == "Analysis":

    if not st.session_state.model_trained:
        st.warning("Train model first")
    else:

        st.subheader("Accuracy Curve")

        fig = go.Figure()
        fig.add_trace(go.Scatter(y=st.session_state.acc_history))
        st.plotly_chart(fig)

        st.subheader("Attention Evolution")

        att = np.array(st.session_state.att_history)
        for i in range(4):
            fig.add_trace(go.Scatter(y=att[:,i], name=f"Mod {i}"))
        st.plotly_chart(fig)

        st.subheader("Confusion Matrix")

        cm = confusion_matrix(
            st.session_state.y_test,
            st.session_state.y_pred
        )

        st.write(cm)

        st.subheader("Report")

        report = classification_report(
            st.session_state.y_test,
            st.session_state.y_pred,
            output_dict=True
        )

        st.dataframe(pd.DataFrame(report).transpose())
