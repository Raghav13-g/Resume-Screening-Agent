import tempfile
from pathlib import Path
import pdfplumber
import docx2txt

def parse_pdf(path):
    text = ""
    with pdfplumber.open(path) as pdf:
        for p in pdf.pages:
            text += p.extract_text() or ""
    return text

def parse_docx(path):
    return docx2txt.process(path)

def extract_text_from_uploaded(file):
    suffix = Path(file.name).suffix.lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(file.read())
        tmp.flush()
        p = tmp.name
    if suffix == ".pdf":
        return parse_pdf(p)
    elif suffix in [".doc", ".docx"]:
        return parse_docx(p)
    else:
        try:
            with open(p, "r", encoding="utf-8") as f:
                return f.read()
        except:
            return ""
