import os
import csv
from datetime import datetime
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from duckduckgo_search import DDGS

# Google Ka Naya SDK (Jo 404 error nahi dega)
from google import genai
from google.genai import types

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Aapki Purani API Key yahan perfectly chalegi
GEMINI_KEY = os.environ.get("GEMINI_API_KEY")

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

def fetch_internet_data(query: str):
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
        raise HTTPException(status_code=500, detail="API Key is missing on server.")

    # Naya Client Setup
    client = genai.Client(api_key=GEMINI_KEY)

    current_date = datetime.now().strftime("%A, %d %B %Y")
    last_msg = request.messages[-1].content.lower()
    
    search_triggers = ["latest", "news", "today", "aaj", "report", "update", "current", "who is", "kaun", "price"]
    live_context = fetch_internet_data(last_msg) if any(w in last_msg for w in search_triggers) else ""

    # XONEXA-AGI MASTER RULES
    SYSTEM_PROMPT = f"""
    You are XONEXA-AGI, an elite Data Intelligence AI by ASTRIX-S (Founder: Vikash Kumar).
    Date: {current_date}
    
    CORE RULES:
    1. NO BS / EXTREMELY DIRECT: Keep normal conversations to 1-3 sentences.
    2. SMART DATA USE: Use the following live internet data if available: {live_context}
    3. BEAUTIFUL FORMATTING: Use Markdown tables and clean formatting.
    4. DATA SCIENCE EXPERT: Agar user data cleaning, data labeling software, data analysis ya image chart ke liye code mangta hai, toh BINA FALTU BAKWASH KIYE sirf Python code do. Uske baad code ko samjhao aur execute karne ka step-by-step tarika batao.
    5. MIT PROFESSOR TONE: Explain logically and simply.
    """

    try:
        formatted_contents = []
        for msg in request.messages:
            role = "model" if msg.role == "assistant" else "user"
            formatted_contents.append(
                types.Content(role=role, parts=[types.Part.from_text(text=msg.content)])
            )
            
        # Sabse advanced aur stable model (gemini-2.0-flash)
        response = client.models.generate_content(
            model='gemini-2.0-flash',
            contents=formatted_contents,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                temperature=0.7,
                max_output_tokens=2048
            )
        )
        return {"reply": response.text}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Google API Error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
