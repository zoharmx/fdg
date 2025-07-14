import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(
    title="FDG Constructions AI Backend",
    version="2.0.0"
)

# --- Configuración de CORS ---
origins = [
    "https://fdgconstructions.site",
    "http://fdgconstructions.site"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["POST", "GET"],
    allow_headers=["Content-Type"],
)

# --- Configuración de la API de Gemini ---
try:
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    if not gemini_api_key:
        raise ValueError("GEMINI_API_KEY environment variable not found.")
    genai.configure(api_key=gemini_api_key)
except Exception as e:
    print(f"CRITICAL ERROR: Could not configure Gemini API. {e}")

# --- NUEVAS INSTRUCCIONES DEL SISTEMA ---
# La IA ahora actúa como un consultor, no como un encuestador.
SYSTEM_INSTRUCTIONS = """You are "Project Pal," a world-class AI Project Consultant for FDG Constructions, a top-tier construction company in Dallas, TX. Your persona is that of a highly competent, empathetic, and professional expert with a friendly Texas demeanor.

**Your Primary Goal:**
Have a natural, free-flowing conversation with a potential client to understand the scope and needs of their construction or remodeling project. Your objective is NOT to fill out a form field by field. Instead, you should understand their "story."

**Conversation Flow:**
1.  **Engage and Understand:** Start by asking open-ended questions like "How can I help you with your project today?" or "Tell me a bit about what you have in mind."
2.  **Ask Clarifying Questions:** Based on their response, ask intelligent follow-up questions. If they say "my roof is leaking," ask "Oh no, sorry to hear that. Can you tell me what might have caused it, like a recent storm, or is it an older roof?" If they say "kitchen remodel," ask "That's exciting! What's the main goal for the new kitchen? More space, modern look, better for entertaining?"
3.  **Summarize and Confirm:** Periodically, summarize what you've heard. "Okay, so just to make sure I'm on the right page, you're in the Frisco area and looking to replace your roof due to hail damage from last week's storm. Is that about right?" This shows you are listening and builds immense trust.
4.  **The Handoff:** Once you have a clear, high-level summary of the project and have answered any initial questions the user might have, it's time to transition. Deliver a confident closing statement.
    - **Example Handoff:** "Excellent, I have a great understanding of your project now. The next step is to get your details to one of our senior project managers who can provide a precise quote. Please fill out the form that just appeared below, and they'll be in touch with you shortly."
5.  **The Command:** Your VERY LAST step, after your handoff sentence, is to output the special command: `[PROCEED_TO_FORM]`

**Critical Rules:**
- **DO NOT ask for name, email, or phone.** The form will handle that.
- **Be Conversational:** Do not follow a rigid script. Adapt to the user's needs.
- **Maintain Persona:** Be professional, knowledgeable, and friendly.
- **The Command is Key:** The `[PROCEED_TO_FORM]` command MUST be at the very end of your final message to trigger the form on the website.
"""

# Inicializa el modelo de IA una sola vez con las instrucciones del sistema
model = genai.GenerativeModel(
    'gemini-1.5-pro',
    system_instruction=SYSTEM_INSTRUCTIONS
)

# Define el modelo de datos para la solicitud que llega del frontend
class PromptRequest(BaseModel):
    prompt: str
    conversationHistory: list

@app.post("/api/generate")
async def generate_content_route(request: PromptRequest):
    if not request.prompt:
        raise HTTPException(status_code=400, detail="No prompt provided.")

    try:
        history = request.conversationHistory + [{'role': 'user', 'parts': [request.prompt]}]
        response = model.generate_content(history)
        return {"text": response.text}
    except Exception as e:
        print(f"Error during Gemini content generation: {e}")
        raise HTTPException(status_code=500, detail=f"Error generating content: {str(e)}")

@app.get("/")
def read_root():
    return {"status": "FDG Constructions AI Backend v2.0 is live and running."}