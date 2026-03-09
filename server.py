import os
import time
from datetime import datetime
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import google.generativeai as genai
from duckduckgo_search import DDGS

app = FastAPI()

# CORS Setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Gemini API Setup
GEMINI_KEY = os.environ.get("GEMINI_API_KEY")
if GEMINI_KEY:
    genai.configure(api_key=GEMINI_KEY)

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: list[Message]

# Smart Internet Search Function
def fetch_latest_data(query: str):
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=3))
            if not results: return ""
            info = "\n[LIVE INTERNET DATA FOUND]:\n"
            for r in results:
                info += f"- {r['title']}: {r['body']}\n"
            return info
    except Exception as e:
        print(f"Search Error: {e}")
        return ""

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    if not GEMINI_KEY:
        raise HTTPException(status_code=500, detail="API Key missing in Render settings.")

    # Live Clock & Date
    current_date = datetime.now().strftime("%A, %d %B %Y")
    current_time = datetime.now().strftime("%I:%M %p")
    
    last_user_msg = request.messages[-1].content.lower()
    
    # Decide if internet search is needed
    search_triggers = ["latest", "news", "today", "aaj", "report", "current", "update", "who is", "kaun hai", "price"]
    live_context = ""
    if any(word in last_user_msg for word in search_triggers):
        live_context = fetch_latest_data(last_user_msg)

    # MASTER SYSTEM PROMPT
    SYSTEM_INSTRUCTION = f"""
    You are XONEXA-AGI, an elite Data Intelligence AI by ASTRIX-S Space Company (Founder: Vikash Kumar).
    
    TODAY'S INFO:
    - Date: {current_date}
    - Time: {current_time}
    
    CORE RULES:
    1. ZERO BORING TALK: Be conversational but extremely direct. 1-3 sentences max for general chat.
    2. SMART SEARCH: Use the [LIVE INTERNET DATA] below if provided to give 100% accurate facts.
    3. MIT STYLE: If asked to explain complex topics, explain like an MIT Professor—logical and simple.
    4. DATA EXPERT: If asked for Data Science, Labeling, or Analysis code, provide ONLY the Python code with clear execution steps.
    5. ENGAGEMENT: Always end with a short, smart question to keep the user interested.
    
    {live_context}
    """

    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        # Build History
        history = [{"role": "user", "parts": [SYSTEM_INSTRUCTION]}]
        for msg in request.messages:
            role = "model" if msg.role == "assistant" else "user"
            history.append({"role": role, "parts": [msg.content]})

        response = model.generate_content(history)
        return {"reply": response.text}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
