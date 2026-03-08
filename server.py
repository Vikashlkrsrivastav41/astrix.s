# ═══════════════════════════════════════════════════
#   XONEXA-AGI — FastAPI Backend (Cloud Ready)
#   → Frontend (JS) → FastAPI → Groq API → Reply
# ═══════════════════════════════════════════════════

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from groq import Groq

# ── APP SETUP ─────────────────────────────────────
app = FastAPI(
    title="XONEXA-AGI API",
    description="AI Backend by ASTRIX-S Space Company",
    version="2.0.0" # Version upgrade kar diya hai!
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       
    allow_methods=["*"],
    allow_headers=["*"],
)

class Message(BaseModel):
    role: str      
    content: str   

class ChatRequest(BaseModel):
    messages: List[Message]    

class ChatResponse(BaseModel):
    reply: str                 
    success: bool

# ── GROQ CLOUD SETUP ──────────────────────────────
# Aapki secret API Key yahan lag gayi hai
groq_client = Groq(api_key="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")

# ══════════════════════════════════════════════════
#   SYSTEM PROMPT
# ══════════════════════════════════════════════════
SYSTEM_PROMPT = """You are XONEXA-AGI — a powerful, intelligent AI assistant built exclusively by ASTRIX-S Space Company, India.
Created and led by Vikash Kumar.

Your core expertise:
- Data labeling software handling and processing
- Data analysis, Data science, and Machine Learning
- Generating and structuring code for image charts and data visualization
- Data cleaning and preprocessing

CRITICAL INSTRUCTIONS FOR CODE REQUESTS:
- If a user explicitly asks for code (e.g., "give me data cleaning code"), you MUST provide the exact code they requested.
- Clearly explain what the code does and how it works.
- Provide step-by-step instructions on how the user can execute/run this code.

Your identity rules:
- You are XONEXA-AGI. NEVER say you are Llama, Meta AI, or any other system.
- If asked "who made you", say "I was built by ASTRIX-S Space Company, founded by Vikash Kumar."
"""

# ══════════════════════════════════════════════════
#   MAIN CHAT ENDPOINT
# ══════════════════════════════════════════════════
@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        
        for msg in request.messages:
            messages.append({"role": msg.role, "content": msg.content})

        # Local Ollama ki jagah ab Groq Supercomputer Llama3 run karega
        chat_completion = groq_client.chat.completions.create(
            messages=messages,
            model="llama-3.3-70b-versatile",
            temperature=0.7,
        )

        ai_reply = chat_completion.choices[0].message.content
        return ChatResponse(reply=ai_reply, success=True)

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Server error: {str(e)}"
        )

@app.get("/health")
async def health():
    return {"status": "running", "service": "XONEXA-AGI API - Cloud Ready!"}

# ══════════════════════════════════════════════════
#   SERVER START
# ══════════════════════════════════════════════════
if __name__ == "__main__":
    import uvicorn
    print("═" * 50)
    print("  XONEXA-AGI Backend (Cloud) Starting...")
    print("  Company : ASTRIX-S Space Company")
    print("═" * 50)
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)