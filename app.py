import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import re
import string
import nltk
nltk.download("stopwords", quiet=True)
from nltk.corpus import stopwords

# ===== Preprocessing (khớp với lúc train) =====
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'http\S+|www\.\S+', '', text)
    text = re.sub(r'\d+', '', text)
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = text.split()
    tokens = [w for w in tokens if w not in stop_words and len(w) > 1]
    return ' '.join(tokens)

# ===== Load model =====
@st.cache_resource
def load_models():
    models = {
        "🤖 Logistic Regression": joblib.load("phanloaiemail.pkl"),
        "⚡ SVM": joblib.load("svm_model.pkl"),
        "📐 Naive Bayes": joblib.load("naive_bayes_model.pkl"),
    }
    vectorizer = joblib.load("tfidf_vectorizer.pkl")
    return models, vectorizer

models, vectorizer = load_models()

# ===== Helper: predict cho scratch models =====
def predict_single(model, text, model_name):
    """
    predict_proba của LogisticRegressionM trả array 1D shape (n_samples,)
    mỗi giá trị là P(spam). Lấy [0] để ra scalar.
    """
    clean = preprocess_text(text)
    vec = vectorizer.transform([clean]).toarray()

    label = int(model.predict(vec)[0])

    conf = None
    if hasattr(model, "predict_proba") and "Logistic" in model_name:
        spam_prob = float(np.asarray(model.predict_proba(vec)).flat[0])  # an toàn mọi shape
        prob = spam_prob if label == 1 else (1 - spam_prob)
        conf = f"{prob*100:.1f}%"

    return label, conf

def predict_batch(model, texts, model_name):
    cleaned = [preprocess_text(t) for t in texts]
    vecs = vectorizer.transform(cleaned).toarray()
    labels = model.predict(vecs)

    confidences = None
    if hasattr(model, "predict_proba") and "Logistic" in model_name:
        # predict_proba trả array 1D shape (n_samples,) — P(spam) cho mỗi sample
        spam_probs = np.asarray(model.predict_proba(vecs)).flatten()
        confidences = [
            f"{p*100:.1f}%" if lbl == 1 else f"{(1-p)*100:.1f}%"
            for p, lbl in zip(spam_probs, labels)
        ]

    return labels, confidences

# ===== Page Config =====
st.set_page_config(
    page_title="Spam Email Classifier Pro",
    page_icon="🚨",
    layout="wide"
)

# ===== Card Component =====
def card(title, desc, icon="📌"):
    st.markdown(
        f"""
        <div style="
            padding:1rem;
            border-radius:12px;
            background:#e5e5e5;
            margin-bottom:1rem;
            box-shadow:0 2px 6px rgba(0,0,0,0.08);
        ">
            <h3 style="margin:0;">{icon} {title}</h3>
            <p style="margin:0.2rem 0 0.6rem 0; color:#444;">{desc}</p>
        </div>
        """,
        unsafe_allow_html=True
    )

st.markdown("<h1 style='text-align:center;'>🚨 Spam Email Classifier</h1>", unsafe_allow_html=True)

# ===== Tabs =====
tab1, tab2, tab3 = st.tabs(["🏠 Dashboard", "📝 Test Email", "📂 Batch Upload"])

# ===== Dashboard =====
with tab1:
    st.info("Ứng dụng phân loại Spam/Ham với TF-IDF và 3 model tự xây dựng.")

    col1, col2, col3 = st.columns(3)
    with col1:
        card("Logistic Regression", "Tự implement từ scratch, gradient descent", icon="🤖")
    with col2:
        card("SVM", "Linear SVM tự xây dựng với hinge loss", icon="⚡")
    with col3:
        card("Naive Bayes", "Gaussian Naive Bayes tự viết", icon="📐")

    with st.expander("📈 Ví dụ phân bố Spam/Ham"):
        sample = pd.DataFrame({"Type": ["Spam", "Ham"], "Count": [60, 40]})
        col_A, col_B, col_C = st.columns([1, 2, 1])
        with col_B:
            fig_d, ax_d = plt.subplots()
            sns.barplot(data=sample, x="Type", y="Count", ax=ax_d, hue="Count", palette="coolwarm")
            st.pyplot(fig_d, use_container_width=True)

# ===== Test Email =====
with tab2:
    st.subheader("✉️ Test Email Realtime")

    selected_model_name = st.selectbox("🔧 Chọn model:", list(models.keys()), key="tab2_model")
    model = models[selected_model_name]

    review = st.text_area("✍️ Nhập nội dung email:", height=150)

    if st.button("🚀 Phân loại"):
        if review.strip():
            label, conf = predict_single(model, review, selected_model_name)

            st.write("### 🔎 Kết quả:")
            conf_str = f" (Confidence: {conf})" if conf else ""

            if label == 1:
                st.error(f"💀 Spam Detected!{conf_str}")
            else:
                st.success(f"✅ Safe (Ham){conf_str}")

            if conf is None:
                st.caption("ℹ️ Model này không hỗ trợ xác suất tin cậy.")

            if label == 1:
                keywords = ["free", "click", "win", "offer", "verify", "account",
                            "login", "secure", "update", "payment"]
                highlighted = review
                for k in keywords:
                    highlighted = re.sub(
                        rf'\b{k}\b',
                        f"<mark style='background:red;color:white;'>{k}</mark>",
                        highlighted, flags=re.IGNORECASE
                    )
                st.markdown(f"### 📌 Highlighted Email\n{highlighted}", unsafe_allow_html=True)
        else:
            st.warning("⚠️ Vui lòng nhập nội dung email!")

# ===== Batch Upload =====
with tab3:
    with st.expander("📂 Upload file CSV"):
        file_upload = st.file_uploader("Chọn file CSV", type=["csv"])

    selected_model_name_b = st.selectbox("🔧 Chọn model:", list(models.keys()), key="tab3_model")
    model_b = models[selected_model_name_b]

    if file_upload is not None:
        data = pd.read_csv(file_upload).dropna().drop_duplicates()
        data = data[data["Category"].isin(['ham', 'spam'])]

        if "Message" not in data.columns:
            st.error("❌ File phải có cột 'Message'")
        else:
            y_pred, confidences = predict_batch(model_b, data["Message"].tolist(), selected_model_name_b)

            data["Prediction"] = ["Spam" if p == 1 else "Ham" for p in y_pred]
            data["Confidence"] = confidences if confidences is not None else "N/A"

            st.success("✅ Phân loại thành công!")

            with st.expander("📊 Xem kết quả dự đoán chi tiết"):
                st.dataframe(data[["Message", "Prediction", "Confidence"]])

            y_test = data["Category"].map({'ham': 0, 'spam': 1})
            cm = confusion_matrix(y_test, y_pred)

            with st.expander("📊 Xem đánh giá mô hình"):
                col_A, _, col_B = st.columns([5, 1, 5])
                with col_A:
                    fig_, ax_ = plt.subplots()
                    sns.countplot(data=data, x="Prediction", hue="Prediction",
                                  palette="coolwarm", ax=ax_)
                    ax_.legend(title="Loại Email", labels=["Ham", "Spam"],
                               loc="upper right", fontsize=10, frameon=True)
                    ax_.set_title("Phân bố Spam/Ham")
                    st.pyplot(fig_, use_container_width=True)

                with col_B:
                    fig, ax = plt.subplots()
                    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                                xticklabels=["Ham", "Spam"],
                                yticklabels=["Ham", "Spam"], ax=ax)
                    ax.set_xlabel("Predicted")
                    ax.set_ylabel("Actual")
                    ax.set_title(f"Confusion Matrix — {selected_model_name_b}")
                    st.pyplot(fig, use_container_width=True)

            csv = data.to_csv(index=False).encode("utf-8")
            st.download_button("📥 Download kết quả", csv,
                               "spam_predictions.csv", "text/csv",
                               key="download-csv")
