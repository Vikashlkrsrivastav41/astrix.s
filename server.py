import os
import csv
import logging
from datetime import datetime
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import google.generativeai as genai
from duckduckgo_search import DDGS

# --- 1. INDUSTRY LEVEL LOGGING ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("XONEXA-AGI")

app = FastAPI(title="XONEXA-AGI Production Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 2. API KEY CHECK ---
GEMINI_KEY = os.environ.get("GEMINI_API_KEY")
if GEMINI_KEY:
    genai.configure(api_key=GEMINI_KEY)
else:
    logger.warning("CRITICAL: GEMINI_API_KEY is not set!")

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

# --- 3. LEAD SAVING SYSTEM ---
@app.post("/contact")
async def save_lead(lead: Lead):
    try:
        with open("leads.csv", "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([lead.name, lead.email, lead.interest, lead.message])
        return {"status": "success", "message": "Data saved!"}
    except Exception as e:
        logger.error(f"Lead Save Error: {e}")
        raise HTTPException(status_code=500, detail="Database Error")

# --- 4. INTERNET AGENT (Auto Research) ---
def fetch_internet_agent(query: str):
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=3))
            if not results: return ""
            info = "\n[LIVE INTERNET DATA]:\n"
            for r in results: info += f"- {r['title']}: {r['body']}\n"
            return info
    except Exception as e:
        logger.error(f"Search Agent Error: {e}")
        return ""

# --- 5. CORE AI LOGIC ---
@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    if not GEMINI_KEY:
        raise HTTPException(status_code=500, detail="API Key missing on server.")

    current_date = datetime.now().strftime("%A, %d %B %Y")
    last_msg = request.messages[-1].content.lower()
    
    # Auto-trigger Search
    search_triggers = ["latest", "news", "today", "aaj", "report", "update", "current", "who is", "kaun", "price"]
    live_context = fetch_internet_agent(last_msg) if any(w in last_msg for w in search_triggers) else ""

    # MASTER RULES (Data Science & Labeling override added)
    SYSTEM_PROMPT = f"""
    You are XONEXA-AGI, an elite Data Intelligence AI by ASTRIX-S (Founder: Vikash Kumar).
    Date: {current_date}
    
    CORE RULES:
    1. NO BS / EXTREMELY DIRECT: Keep normal conversations to 1-3 sentences.
    2. INTERNET AGENT: Use the following live internet data if available: {live_context}
    3. BEAUTIFUL FORMATTING: Use Markdown tables and clean formatting.
    4. DATA SCIENCE EXPERT: If the user asks for code for data cleaning, data labeling, data analysis, or generating image charts, provide ONLY the Python code first. Then explain how to execute it step-by-step. Act as the core intelligence for ASTRIX-S software.
    5. MIT PROFESSOR TONE: Explain logically and simply.
    """

    # --- 6. MEMORY SYSTEM ---
    formatted_history = [
        {"role": "user", "parts": [SYSTEM_PROMPT]},
        {"role": "model", "parts": ["Understood. I am XONEXA-AGI. Ready to deploy."]}
    ]
    
    for msg in request.messages:
        role = "model" if msg.role == "assistant" else "user"
        formatted_history.append({"role": role, "parts": [msg.content]})

    # --- 7. AUTO MODEL FALLBACK (Zero API Crash) ---
    # Sabse pehle aapka bataya hua sabse fast aur naya model
    stable_models = [
        "gemini-2.0-flash",     # The new King
        "gemini-1.5-pro",       # Heavy lifting backup
        "gemini-1.5-flash"      # Ultimate fallback
    ]
    
    last_error = ""
    for model_name in stable_models:
        try:
            model = genai.GenerativeModel(model_name)
            response = model.generate_content(
                formatted_history,
                generation_config={
                    "temperature": 0.7,
                    "max_output_tokens": 2048
                }
            )
            return {"reply": response.text}
        except Exception as e:
            logger.warning(f"Model {model_name} failed: {e}")
            last_error = str(e)
            continue # Ek fail hoga toh automatic dusra chalega bina user ko bataye
            
    # Agar sab fail ho jaye (Jo namumkin hai)
    logger.error(f"XONEXA CORE CRASH: All models failed. Error: {last_error}")
    raise HTTPException(status_code=500, detail=f"Google API Error: {last_error}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
