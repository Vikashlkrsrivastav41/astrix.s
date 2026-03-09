import os
import csv
from datetime import datetime
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import google.generativeai as genai
from duckduckgo_search import DDGS

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

GEMINI_KEY = os.environ.get("GEMINI_API_KEY")
if GEMINI_KEY:
    genai.configure(api_key=GEMINI_KEY)

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: list[Message]

class Lead(BaseModel):
    name: str
    email: str
    interest: str
    message: str

@app.post("/contact")
async def save_lead(lead: Lead):
    try:
        with open("leads.csv", "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([lead.name, lead.email, lead.interest, lead.message])
        return {"status": "success", "message": "Data saved!"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def fetch_latest_data(query: str):
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=3))
            if not results: return ""
            info = "\n[LIVE INTERNET DATA]:\n"
            for r in results: info += f"- {r['title']}: {r['body']}\n"
            return info
    except:
        return ""

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    if not GEMINI_KEY:
        raise HTTPException(status_code=500, detail="API Key is missing.")

    current_date = datetime.now().strftime("%A, %d %B %Y")
    last_msg = request.messages[-1].content.lower()
    
    search_triggers = ["latest", "news", "today", "aaj", "report", "update", "current", "who is", "kaun", "price"]
    live_context = fetch_latest_data(last_msg) if any(w in last_msg for w in search_triggers) else ""

    SYSTEM_PROMPT = f"""
    You are XONEXA-AGI, an elite Data Intelligence AI by ASTRIX-S (Founder: Vikash Kumar).
    Date: {current_date}
    
    CORE RULES:
    1. NO BS / EXTREMELY DIRECT: Keep normal conversations to 1-3 sentences.
    2. SMART DATA USE: Use the following live internet data if available: {live_context}
    3. BEAUTIFUL FORMATTING: Use Markdown tables and clean formatting.
    4. DATA SCIENCE & LABELING EXPERT: If the user asks for code, provide ONLY the pure Python code first. Then explain it step-by-step.
    5. MIT PROFESSOR TONE: Explain logically and simply.
    """

    formatted_history = []
    for i, msg in enumerate(request.messages):
        role = "model" if msg.role == "assistant" else "user"
        content = msg.content
        
        if i == len(request.messages) - 1 and role == "user":
            content = f"{SYSTEM_PROMPT}\n\nUser Question: {content}"
            
        formatted_history.append({"role": role, "parts": [content]})
        
    # --- THE MAGIC LOOP: Yeh Google ke nakhre khatam kar dega ---
    models_to_try = [
        'gemini-1.5-flash-latest', 
        'gemini-1.5-flash', 
        'gemini-1.0-pro-latest',
        'gemini-1.0-pro',
        'gemini-pro'
    ]
    
    last_error_message = ""
    for model_name in models_to_try:
        try:
            model = genai.GenerativeModel(model_name)
            response = model.generate_content(formatted_history)
            return {"reply": response.text} # Jaise hi koi ek chalega, yahi se reply de dega
        except Exception as e:
            last_error_message = str(e)
            continue # Agar fail hua, toh agla model try karega
            
    # Agar sab fail ho gaye (jo ki namumkin hai)
    raise HTTPException(status_code=500, detail=f"Google API Error: {last_error_message}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
