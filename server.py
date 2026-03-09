import os
import csv
from datetime import datetime
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import google.generativeai as genai
from duckduckgo_search import DDGS

app = FastAPI()

# CORS Setup (Frontend ko connect karne ke liye)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Gemini Engine Setup
GEMINI_KEY = os.environ.get("GEMINI_API_KEY")
if GEMINI_KEY:
    genai.configure(api_key=GEMINI_KEY)

# Data Models
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

# 1. DATA SAVE LOGIC (Leads)
@app.post("/contact")
async def save_lead(lead: Lead):
    try:
        with open("leads.csv", "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([lead.name, lead.email, lead.interest, lead.message])
        return {"status": "success", "message": "Data saved!"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 2. SMART INTERNET SEARCH LOGIC
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

# 3. MAIN AI CHAT LOGIC
@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    if not GEMINI_KEY:
        raise HTTPException(status_code=500, detail="API Key is missing in Render Environment Variables.")

    last_msg = request.messages[-1].content.lower()
    
    # Auto-detect if internet search is needed
    search_triggers = ["latest", "news", "today", "aaj", "report", "update", "current", "who is", "kaun", "price"]
    live_context = ""
    if any(word in last_msg for word in search_triggers):
        live_context = fetch_latest_data(last_msg)

    current_date = datetime.now().strftime("%A, %d %B %Y")

    # 4. MASTER MIND PROMPT (XONEXA-AGI RULES)
    SYSTEM_PROMPT = f"""
    You are XONEXA-AGI, an elite Data Intelligence AI by ASTRIX-S (Founder: Vikash Kumar).
    Today's Date: {current_date}
    
    CORE RULES:
    1. NO BS / EXTREMELY DIRECT: Keep normal conversations to 1-3 sentences. Be highly engaging and addictive.
    2. SMART DATA USE: Use the following live internet data if available: {live_context}
    3. BEAUTIFUL FORMATTING: Always use Markdown tables when comparing items, listing features, or organizing data. Use simple text-based diagrams or structured bullet points to explain concepts.
    4. DATA SCIENCE & LABELING EXPERT: If the user asks for code (e.g., for data cleaning, data labeling, data analysis, or generating image charts), provide ONLY the pure Python code first. Then, clearly explain how the code works, what it does, and provide step-by-step execution instructions. Act as the core intelligence for ASTRIX-S data labeling software.
    5. MIT PROFESSOR TONE: Explain complex things logically and simply.
    6. ALWAYS ENGAGE: End your response with a highly relevant, smart follow-up question.
    """

    # Error-proof Model Execution
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        history = [{"role": "user", "parts": [SYSTEM_PROMPT]}]
        
        for msg in request.messages:
            role = "model" if msg.role == "assistant" else "user"
            history.append({"role": role, "parts": [msg.content]})
            
        response = model.generate_content(history)
        return {"reply": response.text}
        
    except Exception as e:
        print(f"Primary model failed, trying fallback: {e}")
        try:
            # Fallback model if the first one throws a version error
            backup_model = genai.GenerativeModel('gemini-pro')
            response = backup_model.generate_content(history)
            return {"reply": response.text}
        except Exception as inner_e:
            raise HTTPException(status_code=500, detail=str(inner_e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
