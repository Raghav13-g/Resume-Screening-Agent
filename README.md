# 🚀 Resume Screening Agent

The **Resume Screening Agent** is an AI-powered tool that automatically reads resumes, analyzes job descriptions, and ranks candidates based on how well they match.  
It helps recruiters and hiring teams save time by generating match scores, identifying key skills, and highlighting missing skills.

---

## ✅ Features

- Reads & processes resumes (PDF/Text)
- Understands job descriptions using NLP
- Computes vector similarity using Sentence Transformers
- Generates Match Score (%)
- Shows Matched Skills & Missing Skills
- Ranks candidates automatically

---

## 🧠 Tech Stack

- **Python 3.10+**
- **Sentence Transformers** (MPNet / MiniLM fallback)
- **NumPy & Scikit-Learn**
- **(Optional)** OpenAI / Gemini LLM integration
- **Vector Similarity Search**

---

## 📁 Project Structure
Resume-Screening-Agent/
│
- ├── app.py # Main script / entrypoint
- ├── vector_store.py # Embedding model + similarity logic
- ├── utils.py # Parsing + helper utilities
- ├── requirements.txt # Python dependencies
- └── README.md # This file


---

## ⚙️ How it works (simple)
1. Load resumes (PDF/text) and a job description.  
2. Convert documents into embeddings (vectors).  
3. Compute vector similarity between JD and each resume.  
4. Produce a ranked list of candidates with match scores and skill reports.

---

## 📦 Installation

1. Clone the repository:
```bash
git clone https://github.com/Raghav13-g/Resume-Screening-Agent.git
cd Resume-Screening-Agent
Create a virtual environment (recommended) and activate it:

python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate


Install dependencies:

pip install -r requirements.txt

