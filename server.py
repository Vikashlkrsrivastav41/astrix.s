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
    4. DATA SCIENCE & LABELING EXPERT: If the user asks for code (e.g., data cleaning, labeling, generating image charts), provide ONLY the pure Python code first. Then, clearly explain how the code works and provide execution instructions. Act as the core intelligence for ASTRIX-S software.
    5. MIT PROFESSOR TONE: Explain logically and simply.
    """

    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        # PERFECT HISTORY LOGIC (Error yahi theek hua hai)
        formatted_history = []
        for i, msg in enumerate(request.messages):
            role = "model" if msg.role == "assistant" else "user"
            content = msg.content
            
            # System Prompt ko sirf aakhri user message ke andar chupke se bhej rahe hain
            if i == len(request.messages) - 1 and role == "user":
                content = f"{SYSTEM_PROMPT}\n\nUser Question: {content}"
                
            formatted_history.append({"role": role, "parts": [content]})
            
        response = model.generate_content(formatted_history)
        return {"reply": response.text}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
