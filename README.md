# AI-Powered Resume Ranker (Streamlit)

Match resumes to a Job Description using NLP and BERT embeddings.  
Upload multiple PDFs, paste a JD, and get ranked results with similarity scores and a bar chart.

## 🔧 Tech
- Streamlit, scikit-learn, spaCy, NLTK, SentenceTransformers (BERT), Plotly

## 🚀 Run locally
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
streamlit run app.py
