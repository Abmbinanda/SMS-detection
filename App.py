import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Page Configuration
st.set_page_config(
    page_title="Spam Detection System",
    page_icon="📩",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>

/* Main Container */
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 1200px;
}

/* Title */
.main-title {
    text-align: center;
    font-size: 2.8rem;
    font-weight: 700;
    margin-bottom: 10px;
}

/* Card */
.card {
    background-color: #ffffff;
    padding: 20px;
    border-radius: 15px;
    border: 1px solid #e5e7eb;
}

/* Buttons */
.stButton > button {
    width: 100%;
    height: 55px;
    font-size: 18px;
    font-weight: bold;
    border-radius: 12px;
}

/* Metrics */
[data-testid="metric-container"] {
    border: 1px solid #e5e7eb;
    border-radius: 12px;
    padding: 15px;
}

/* Mobile */
@media (max-width:768px) {

    .main-title {
        font-size: 2rem;
    }

    .block-container {
        padding-left: 1rem;
        padding-right: 1rem;
    }

    .stButton > button {
        height: 50px;
        font-size: 16px;
    }
}
</style>
""", unsafe_allow_html=True)

# Load Model
import os
import pickle

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

import os
import pickle

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

print("Current directory:", os.getcwd())
print("App directory:", BASE_DIR)

tfidf = pickle.load(open(os.path.join(BASE_DIR, "vectorizer.pkl"), "rb"))
model = pickle.load(open(os.path.join(BASE_DIR, "model.pkl"), "rb"))

tfidf = pickle.load(open(os.path.join(BASE_DIR, "vectorizer.pkl"), "rb"))
model = pickle.load(open(os.path.join(BASE_DIR, "model.pkl"), "rb"))

ps = PorterStemmer()

# Title
st.markdown(
    "<div class='main-title'>📩 Spam Detection System</div>",
    unsafe_allow_html=True
)

st.markdown("---")

# Message Type Selection
message_type = st.selectbox(
    "Select Message Type",
    ["SMS", "Email"]
)

# Input Area
if message_type == "SMS":
    user_input = st.text_area(
        "Enter SMS Message",
        height=150,
        placeholder="Type your SMS message here..."
    )
else:
    user_input = st.text_area(
        "Enter Email Content",
        height=200,
        placeholder="Paste email content here..."
    )

# Analyze Button
if st.button("🔍 Analyze Message"):

    if user_input.strip() == "":
        st.warning("Please enter a message.")

    else:

        # Original
        original_text = user_input

        # Lowercase
        lowercase_text = original_text.lower()

        # Tokenization
        tokens = nltk.word_tokenize(lowercase_text)

        # Punctuation Removal
        punctuation_removed = []

        for word in tokens:
            if word.isalnum():
                punctuation_removed.append(word)

        # Stopword Removal
        stopword_removed = []

        for word in punctuation_removed:
            if word not in stopwords.words('english'):
                stopword_removed.append(word)

        # Stemming
        stemmed_words = []

        for word in stopword_removed:
            stemmed_words.append(ps.stem(word))

        # Final Processed Text
        transformed_text = " ".join(stemmed_words)

        # Prediction
        vector_input = tfidf.transform([transformed_text])
        result = model.predict(vector_input)[0]

        st.markdown("---")

        st.subheader("Analysis Result")

        if result == 1:
            st.error("🚨 SPAM DETECTED")
        else:
            st.success("✅ LEGITIMATE MESSAGE")

        st.markdown("---")

        # Statistics
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Characters", len(user_input))

        with col2:
            st.metric("Words", len(user_input.split()))

        with col3:
            st.metric("Type", message_type)

        st.markdown("---")

        st.subheader("NLP Processing Pipeline")

        with st.expander("1️⃣ Original Message"):
            st.write(original_text)

        with st.expander("2️⃣ Lowercase Conversion"):
            st.write(lowercase_text)

        with st.expander("3️⃣ Tokenization"):
            st.write(tokens)

        with st.expander("4️⃣ Punctuation Removal"):
            st.write(punctuation_removed)

        with st.expander("5️⃣ Stopword Removal"):
            st.write(stopword_removed)

        with st.expander("6️⃣ Stemming"):
            st.write(stemmed_words)

        with st.expander("7️⃣ Final Processed Text"):
            st.write(transformed_text)