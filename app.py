import streamlit as st
import joblib

# ============================================================
# Page configuration
# ============================================================
st.set_page_config(
    page_title="Spam Classifier",
    page_icon="📧",
    layout="centered"
)

# ============================================================
# Load the trained model and vectorizer
# ============================================================
@st.cache_resource
def load_artifacts():
    model = joblib.load("spam_email.pkl")
    vectorizer = joblib.load("vectorizer.pkl")
    return model, vectorizer

model, vectorizer = load_artifacts()

# ============================================================
# Header
# ============================================================
st.title("📧 SMS Spam Classifier")
st.write(
    "Type a message below and the model will predict whether it's **spam** or **ham** (legitimate)."
)

# ============================================================
# Disclaimer about model limitations
# ============================================================
st.info(
    """
    ℹ️ **About this model:** Trained on the *SMS Spam Collection* dataset (~2002, British SMS). 
    It performs well on classic spam patterns ("WINNER!!", "FreeMsg", premium-rate numbers) 
    but may miss modern spam (iPhone scams, phishing URLs, account verification cons). 
    This is **distribution shift** — a real ML limitation that production filters address by 
    retraining continuously.
    """
)

# ============================================================
# Example messages
# ============================================================
st.subheader("Try an example")
col1, col2 = st.columns(2)

examples = {
    "📨 Classic spam": "Free entry in 2 a wkly comp to win FA Cup final tkts. Text FA to 87121 to receive entry question",
    "🏆 Prize spam": "WINNER!! As a valued network customer you have been selected to receive a £900 prize reward",
    "👋 Friendly text": "Hey, are we still meeting at 7 tonight? lmk",
    "🍕 Casual ham": "Mom says dinner is ready, come downstairs",
}


if "message_input" not in st.session_state:
    st.session_state.message_input = ""

with col1:
    if st.button(list(examples.keys())[0], use_container_width=True):
        st.session_state.message_input = list(examples.values())[0]
    if st.button(list(examples.keys())[2], use_container_width=True):
        st.session_state.message_input = list(examples.values())[2]

with col2:
    if st.button(list(examples.keys())[1], use_container_width=True):
        st.session_state.message_input = list(examples.values())[1]
    if st.button(list(examples.keys())[3], use_container_width=True):
        st.session_state.message_input = list(examples.values())[3]


st.subheader("Your message")
message = st.text_area(
    "Type or paste a message here:",
    value=st.session_state.message_input,
    height=100,
    label_visibility="collapsed"
)

if st.button("🔍 Predict", type="primary", use_container_width=True):
    if message.strip() == "":
        st.warning("⚠️ Please enter a message first.")
    else:
        message_vec = vectorizer.transform([message])
        prediction = model.predict(message_vec)[0]
        probabilities = model.predict_proba(message_vec)[0]

        # Find the probability of the predicted class
        class_index = list(model.classes_).index(prediction)
        confidence = probabilities[class_index] * 100

        
        st.subheader("Result")

        if prediction == "spam":
            st.error(f"🚨 This looks like **SPAM**")
        else:
            st.success(f"✅ This looks like **HAM** (legitimate)")

        st.metric(
            label="Model confidence",
            value=f"{confidence:.1f}%",
        )

        with st.expander("See probability breakdown"):
            for cls, prob in zip(model.classes_, probabilities):
                st.write(f"**{cls.upper()}:** {prob * 100:.2f}%")


with st.sidebar:
    st.header("About this project")
    st.markdown(
        """
        **Tech stack:**
        - Python
        - scikit-learn
        - Streamlit
        - pandas

        **Model:** Multinomial Naive Bayes  
        **Features:** TF-IDF vectorization  
        **Test accuracy:** 96.9%  
        **Dataset:** SMS Spam Collection (~5,500 messages)

        ---

        Built by **Rahul**.  
        

        🔗 [GitHub repo](https://github.com/idfwyy/email_spam_classifier)
        """
    )


st.markdown("---")
st.caption(
    "Built with Streamlit • Model trained on the SMS Spam Collection dataset"
)   