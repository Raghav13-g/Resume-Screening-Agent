from dotenv import load_dotenv
load_dotenv()

import os
import re
import json
import time
import streamlit as st
import pandas as pd
import google.generativeai as genai

from parsers import extract_text_from_uploaded
from vector_store import ResumeVectorStore
from skill_match import extract_skills, compute_final_score, parse_required_skills

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

def call_gemini_scoring(jd_text, resume_text, max_retries=2, wait_s=1.0):
    prompt = f"""
You are an expert recruiter. Score the candidate 0-100 for fit to the job description and return JSON only like:
{{ "score": <int 0-100>, "justification": "<2-3 line justification>" }}

Job Description:
{jd_text}

Resume:
{resume_text}
"""
    attempt = 0
    last_raw = ""
    while attempt <= max_retries:
        attempt += 1
        try:
            model = genai.GenerativeModel("gemini-2.0-flash")
            resp = model.generate_content(prompt)
            raw = getattr(resp, "text", str(resp))
            last_raw = raw.strip()
            m = re.search(r"\{.*\}", last_raw, re.S)
            if m:
                try:
                    parsed = json.loads(m.group(0))
                    score = int(parsed.get("score", 50))
                    justification = parsed.get("justification", "")[:400]
                    return {"score": score, "justification": justification, "raw": last_raw}
                except Exception:
                    pass
            m2 = re.search(r"score[:\s\-]+(\d{1,3})", last_raw, re.I)
            if m2:
                score = int(m2.group(1))
                justification = re.sub(r"score[:\s\-]+\d{1,3}", "", last_raw, flags=re.I).strip()[:400]
                return {"score": max(0, min(100, score)), "justification": justification, "raw": last_raw}
            m3 = re.search(r"(\d{1,3})", last_raw)
            if m3:
                score = int(m3.group(1))
                justification = last_raw.replace(m3.group(1), "").strip()[:400]
                return {"score": max(0, min(100, score)), "justification": justification, "raw": last_raw}
            return {"score": 50, "justification": last_raw[:400], "raw": last_raw}
        except Exception as e:
            last_raw = f"[LLM error] {e}"
            time.sleep(wait_s)
            continue
    return {"score": 50, "justification": last_raw[:400], "raw": last_raw}

st.set_page_config(page_title="Resume Screening Agent", layout="wide")
st.title("Resume Screening Agent")

with st.sidebar:
    jd_text = st.text_area("Job Description", height=220)
    jd_file = st.file_uploader("Upload JD File", type=["pdf","docx","txt"])
    resumes = st.file_uploader("Upload Resumes", type=["pdf","docx","txt"], accept_multiple_files=True)
    required_skills_input = st.text_input("Required Skills (comma separated)")
    top_k_llm = st.number_input("LLM Top-K", min_value=0, max_value=20, value=5)
    run_btn = st.button("Run Screening")

if jd_file and not jd_text.strip():
    jd_text = extract_text_from_uploaded(jd_file)

if run_btn:
    if not jd_text.strip() or not resumes:
        st.error("Provide a job description and at least one resume.")
        st.stop()

    vs = ResumeVectorStore("./chroma_storage")
    vs.reset_collection()

    for f in resumes:
        text = extract_text_from_uploaded(f)
        vs.add_resume(f.name, text, {"name": f.name})

    results = vs.query(jd_text, n_results=len(resumes))

    if required_skills_input.strip():
        required_skills = parse_required_skills(required_skills_input)
    else:
        required_skills = extract_skills(jd_text)

    rows = []
    for i, r in enumerate(results):
        name = r.get("metadata", {}).get("name") or f"candidate_{i+1}"
        text = r.get("document", "") or ""
        similarity = 1 - r.get("distance", 1.0)
        skills = extract_skills(text)
        if i < top_k_llm:
            score_obj = call_gemini_scoring(jd_text, text)
            llm_score = score_obj.get("score", 50)
            justification = score_obj.get("justification", "")
            raw_llm_output = score_obj.get("raw", "")
        else:
            llm_score = None
            justification = ""
            raw_llm_output = ""
        final_score, used_skills = compute_final_score(similarity, text, required_skills, jd_text, llm_score)
        rows.append({
            "name": name,
            "final_score": round(final_score, 2),
            "similarity": round(similarity, 4),
            "skills": ", ".join(used_skills[:10]),
            "llm_score": llm_score,
            "justification": justification,
            "raw_llm": raw_llm_output
        })

    df = pd.DataFrame(rows).sort_values("final_score", ascending=False).reset_index(drop=True)
    st.success("Screening complete")
    st.dataframe(df[["name","final_score","similarity","llm_score","skills","justification"]])
    st.download_button("Download CSV", df.to_csv(index=False), "screening_results.csv")
    for idx, row in df.iterrows():
        raw = row.get("raw_llm", "")
        if raw:
            with st.expander(f"Raw Gemini output — {row['name']}", expanded=False):
                st.text(raw)
