import PyPDF2
import nltk
import spacy
import re
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

# Download stopwords if not already
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
nlp = spacy.load("en_core_web_sm")

# Function to extract text from PDF
def extract_text_from_pdf(uploaded_file):
    text = ""
    reader = PyPDF2.PdfReader(uploaded_file)
    for page in reader.pages:
        text += page.extract_text() + " "
    return text

# Preprocess text
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    doc = nlp(text)
    tokens = [token.lemma_ for token in doc if token.text not in stop_words and not token.is_punct]
    return " ".join(tokens)

# Ranking function
def rank_resumes(resume_texts, job_description):
    corpus = [job_description] + resume_texts
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(corpus)
    similarity_scores = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
    return similarity_scores

# Streamlit UI
st.title("🤖 AI-Powered Resume Ranker")
jd_input = st.text_area("Paste Job Description Here")

uploaded_files = st.file_uploader("Upload Resumes (PDF)", type="pdf", accept_multiple_files=True)

if st.button("Rank Resumes"):
    resume_texts = []
    for file in uploaded_files:
        resume_text = extract_text_from_pdf(file)
        processed = preprocess_text(resume_text)
        resume_texts.append(processed)

    processed_jd = preprocess_text(jd_input)
    scores = rank_resumes(resume_texts, processed_jd)

    ranked_resumes = sorted(zip(uploaded_files, scores), key=lambda x: x[1], reverse=True)

    st.subheader("📊 Ranking Results")
    for file, score in ranked_resumes:
        st.write(f"**{file.name}** → Similarity Score: {score:.2f}")
