import streamlit as st
import pickle
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()

# ---- SESSION STATE ----
if 'input_sms' not in st.session_state:
    st.session_state.input_sms = ""

if 'result' not in st.session_state:
    st.session_state.result = ""

# ---- RESET FUNCTION ----
def clear_all():
    st.session_state.input_sms = ""
    st.session_state.result = ""

# ---- FUNCTION ----
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

# ---- LOAD ----
tfidf = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))

st.title("📩 Email/SMS Classifier")

# ---- INPUT ----
input_sms = st.text_input("Enter your message", key="input_sms")

# ---- BUTTONS ----
col1, col2 = st.columns(2)

with col1:
    if st.button('Predict'):
        if input_sms.strip() != "":
            transform_sms = transform_text(input_sms)
            vector_input = tfidf.transform([transform_sms])
            result = model.predict(vector_input)[0]

            if result == 1:
                st.session_state.result = "🚨 Spam"
            else:
                st.session_state.result = "✅ Not Spam"

with col2:
    st.button('Reset', on_click=clear_all)   # ✅ correct way

# ---- DISPLAY ----
if st.session_state.result:
    st.header(st.session_state.result)