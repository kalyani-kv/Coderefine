from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from groq import Groq
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.lib.pagesizes import letter
from reportlab.platypus import Preformatted
from reportlab.lib.styles import ParagraphStyle
from fastapi.responses import FileResponse
import uvicorn
import os
import uuid

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
client = Groq(api_key=GROQ_API_KEY)

# -----------------------
# Models
# -----------------------
class CodeRequest(BaseModel):
    code: str
    language: str
    focus: list[str] = []

class TextRequest(BaseModel):
    text: str

class TranslateRequest(BaseModel):
    code: str
    fromLang: str
    toLang: str


# -----------------------
# Utility Function
# -----------------------
def ask_llm(system_prompt, user_prompt, temperature=0.3):
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=temperature,
        max_tokens=1500
    )
    return response.choices[0].message.content


# -----------------------
# REVIEW
# -----------------------
@app.post("/review")
def review_code(request: CodeRequest):
    prompt = f"""
Review this {request.language} code focusing on {', '.join(request.focus)}:

{request.code}
"""
    result = ask_llm("You are a senior code reviewer.", prompt)
    return {"result": result}


# -----------------------
# REWRITE
# -----------------------
@app.post("/rewrite")
def rewrite_code(request: CodeRequest):
    prompt = f"Rewrite this {request.language} code with best practices:\n{request.code}"
    result = ask_llm("You are a professional code rewriter.", prompt, 0.2)
    return {"result": result}


# -----------------------
# DEBUG
# -----------------------
@app.post("/debug")
def debug_code(request: CodeRequest):
    prompt = f"Find bugs and fix this {request.language} code:\n{request.code}"
    result = ask_llm("You are an expert debugger.", prompt)
    return {"result": result}


# -----------------------
# CODE GENERATION
# -----------------------
@app.post("/codegen")
def code_generation(request: CodeRequest):
    prompt = f"Generate {request.language} code for:\n{request.code}"
    result = ask_llm("You are a software architect.", prompt)
    return {"result": result}


# -----------------------
# TEXT GENERATION
# -----------------------
@app.post("/text")
def text_generation(request: TextRequest):
    result = ask_llm("You are a content writer.", request.text)
    return {"result": result}


# -----------------------
# SUMMARIZE
# -----------------------
@app.post("/summarize")
def summarize(request: TextRequest):
    prompt = f"Summarize this text:\n{request.text}"
    result = ask_llm("You are a summarization expert.", prompt)
    return {"result": result}


# -----------------------
# Q&A
# -----------------------
@app.post("/qa")
def qa(request: TextRequest):
    result = ask_llm("Answer the question clearly.", request.text)
    return {"result": result}


# -----------------------
# SENTIMENT
# -----------------------
@app.post("/sentiment")
def sentiment(request: TextRequest):
    prompt = f"Analyze sentiment of:\n{request.text}"
    result = ask_llm("You are a sentiment analysis expert.", prompt)
    return {"result": result}


# -----------------------
# RECOMMEND
# -----------------------
@app.post("/recommend")
def recommend(request: TextRequest):
    prompt = f"Give recommendations for:\n{request.text}"
    result = ask_llm("You are an AI recommendation engine.", prompt)
    return {"result": result}


# -----------------------
# TRANSLATE
# -----------------------
@app.post("/translate")
def translate(request: TranslateRequest):
    prompt = f"""
Translate this code from {request.fromLang} to {request.toLang}:

{request.code}
"""
    result = ask_llm("You are a code translator.", prompt)
    return {"result": result}


# -----------------------
# MULTIMODAL
# -----------------------
@app.post("/multimodal")
def multimodal(request: TextRequest):
    prompt = f"Analyze and explain this input:\n{request.text}"
    result = ask_llm("You are a multimodal AI.", prompt)
    return {"result": result}


# -----------------------
# PDF EXPORT
# -----------------------
@app.post("/pdf")
def generate_pdf(request: TextRequest):
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Empty content")

    filename = f"{uuid.uuid4()}.pdf"
    filepath = os.path.join(os.getcwd(), filename)

    doc = SimpleDocTemplate(filepath, pagesize=letter)
    styles = getSampleStyleSheet()
    elements = []

    elements.append(Preformatted(request.text, styles["Code"]))
    doc.build(elements)

    return FileResponse(
        path=filepath,
        media_type="application/pdf",
        filename="output.pdf"
    )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)