import re
from rapidfuzz import process, fuzz

SKILL_DICT = [
    "python","java","c++","c#","sql","pandas","numpy","scikit-learn","tensorflow",
    "pytorch","keras","spark","hadoop","aws","azure","gcp","docker","kubernetes",
    "rest api","flask","django","react","node.js","javascript","html","css",
    "computer vision","nlp","natural language processing","deep learning","machine learning",
    "etl","powerbi","tableau","bash","linux","git","opencv","devops","data analysis",
    "data science","cloud computing","mongodb","mysql","postgresql"
]

def normalize(t):
    return re.sub(r"\s+", " ", (t or "").lower()).strip()

def extract_skills(text, cutoff=75):
    t = normalize(text)
    found = set(s for s in SKILL_DICT if s in t)

    tokens = list(set(re.findall(r"[a-zA-Z\+\#\.\-]+", t)))
    for tok in tokens:
        match, score, _ = process.extractOne(tok, SKILL_DICT, scorer=fuzz.token_sort_ratio)
        if score >= cutoff:
            found.add(match)

    return sorted(list(found))

def extract_years(text):
    yrs = re.findall(r"(\d{1,2})\s+(years|year|yrs)", text, re.I)
    return max(int(x[0]) for x in yrs) if yrs else 0

def skill_overlap(req, cand):
    if not req:
        return 50
    r = set(s.lower() for s in req)
    c = set(s.lower() for s in cand)
    return int(len(r & c) / len(r) * 100) if r else 50

def parse_required_skills(raw):
    return [x.strip().lower() for x in re.split(r"[,\n;]+", raw) if x.strip()]

def compute_final_score(similarity, text, required_skills, jd_text, llm_score):
    cand_skills = extract_skills(text)
    skill_score = skill_overlap(required_skills, cand_skills)
    yrs = extract_years(text)
    exp_score = min(100, yrs * 10)

    if llm_score is None:
        w_sim, w_skill, w_exp, w_llm = 0.55, 0.35, 0.10, 0.0
    else:
        w_sim, w_skill, w_exp, w_llm = 0.35, 0.35, 0.15, 0.15

    sim_score = similarity * 100
    score = (w_sim * sim_score) + (w_skill * skill_score) + (w_exp * exp_score)

    if llm_score is not None:
        score = score * (1 - w_llm) + llm_score * w_llm

    return score, cand_skills
