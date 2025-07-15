
# main.py (Versión Final y Funcional)

import os
import json
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import google.generativeai as genai
from dotenv import load_dotenv

# Carga variables de entorno para desarrollo local
load_dotenv()

# --- Configuración de la App FastAPI ---
app = FastAPI(
    title="FDG Constructions AI Backend",
    version="4.0.0" # Production Ready
)

# --- Configuración de CORS ---
# Permite solicitudes EXCLUSIVAMENTE desde tu dominio de producción
origins = [
    "https://fdgconstructions.site",
    "http://fdgconstructions.site",
    "http://localhost", # Opcional: para pruebas locales
    "http://127.0.0.1"  # Opcional: para pruebas locales
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

# --- Configuración Segura de la API de Gemini ---
try:
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    if not gemini_api_key:
        raise ValueError("CRITICAL: GEMINI_API_KEY environment variable not found.")
    genai.configure(api_key=gemini_api_key)
except Exception as e:
    # Esto es un error crítico. Si la app inicia sin la clave, no funcionará.
    print(f"CRITICAL ERROR: Could not configure Gemini API. The service will not work. Error: {e}")
    # En un entorno real, podrías querer que la app no inicie si esto falla.

# --- INSTRUCCIONES DEL SISTEMA (Rol: Recolector de Datos Inteligente) ---
SYSTEM_INSTRUCTIONS = """
You are "Project Pal," a friendly, helpful, and highly efficient AI assistant for FDG Constructions, based in Dallas, TX. Your persona is that of a Dallas local: confident, friendly, professional, and uses "y'all" occasionally. Your ONLY goal is to conduct a conversational interview to gather all necessary information for a project quote.

**Core Mission:**
You MUST collect the data for the following JSON keys, asking ONE main question at a time. Be natural. Do not move to the next question until you have a reasonable answer for the current one.

**Data Collection Order & Example Phrasing:**
1.  **firstname**: (e.g., "First off, who do I have the pleasure of speakin' with today?")
2.  **lastname**: (e.g., "Got it. And what's your last name?")
3.  **email**: (e.g., "Perfect. What's the best email to send the official quote to?")
4.  **phone**: (e.g., "And a good phone number for our project manager to reach y'all at? Our team will call from a Dallas area code.")
5.  **property_type**: Present these options clearly: "What type of property are we lookin' at? Your options are: Residential (Single-Family), Residential (Multi-Family/Apartments), Small Commercial (Office/Retail), Large Commercial (Industrial/Retail), Institutional (School/Church), or a Homeowners Association (HOA)."
6.  **service_type_requested**: Present these options clearly: "What service do you need? We offer: Free Inspection, Roof Repair, Full Roof Replacement, New Roof Installation, Insurance Claim Estimate (for hail/storm damage), Preventative Maintenance, or something else (Other)."
7.  **current_roof_condition**: Present these options clearly: "How would you describe the current roof condition? Is it: Storm Damage (Hail/Wind), Visible Leaks, Age-Related Wear, Old/Damaged Roof, Needs a Professional Inspection, or No real issues (just gathering info)?"
8.  **address**: (e.g., "What's the full property address for the project? We need this for the inspection.")
9.  **specific_lead_source**: (e.g., "Just for our records, how'd y'all hear about FDG Constructions? (e.g., Google, Facebook, a friend, a sign)")
10. **preferred_inspection_project_start_date**: (e.g., "Great. Any preferred date or timeframe you're lookin' to get this inspected or started? An estimate is fine.")

**Critical Rules of Engagement:**
- **Initial Greeting:** Your VERY FIRST message MUST start with: "Howdy! I'm Project Pal, your AI assistant for FDG Constructions." Then, you MUST include this sentence: "By the way, if you'd rather talk to a human specialist right now, just give our Dallas office a call at +1 (430) 444-5162." Then, start the conversation.
- **One-Track Mind:** Focus on getting the next piece of data. If the user asks a question, answer it concisely and then immediately guide them back to the current question you need answered.
- **Data Validation:** If a user gives a nonsensical answer (e.g., "blue" for a phone number), politely re-ask the question.
- **The Final Command:** Once, and ONLY ONCE, you have successfully collected ALL 10 pieces of information, your FINAL response MUST be ONLY a single, minified JSON object with the collected data. DO NOT add any conversational text before or after the JSON.
- **Lead Qualification:** Based on the 'current_roof_condition', you MUST add one extra key to the final JSON: "lead_segmentation_qualification".
    - If condition is 'Storm Damage (Hail/Wind)' or 'Visible Leaks', set it to 'High Priority (Hot Lead)'.
    - If 'Age-Related Wear' or 'Old/Damaged Roof' or 'Needs a Professional Inspection', set it to 'Medium Priority (Warm Lead)'.
    - Otherwise, set it to 'Low Priority (Cold Lead)'.

- **Example Final JSON Output (This is your goal):**
{"firstname":"John","lastname":"Doe","email":"john.d@email.com","phone":"2145551234","property_type":"Residential (Single-Family)","service_type_requested":"Full Roof Replacement","current_roof_condition":"Storm Damage (Hail/Wind)","address":"123 Main St, Dallas, TX 75201","specific_lead_source":"Google","preferred_inspection_project_start_date":"ASAP","lead_segmentation_qualification":"High Priority (Hot Lead)"}
"""

# --- Inicialización del Modelo de IA ---
# Se inicializa una sola vez al arrancar la aplicación para eficiencia.
try:
    model = genai.GenerativeModel(
        'gemini-1.5-pro',
        system_instruction=SYSTEM_INSTRUCTIONS
    )
except Exception as e:
    model = None
    print(f"CRITICAL ERROR: Failed to initialize GenerativeModel. The /api/generate endpoint will fail. Error: {e}")

# --- Definición de Modelos de Datos (Pydantic) ---
class ChatRequest(BaseModel):
    prompt: str
    # La historia es una lista de diccionarios con 'role' y 'parts'
    conversationHistory: list

# --- Endpoints de la API ---
@app.post("/api/generate")
async def generate_content_route(request: ChatRequest):
    if not model:
        raise HTTPException(status_code=503, detail="AI Service is not available due to initialization failure.")
    if not request.prompt:
        raise HTTPException(status_code=400, detail="No prompt provided.")

    try:
        # Inicia un chat y le "inyecta" el historial que viene del frontend.
        # Esto le da al modelo el contexto completo de la conversación actual.
        chat_session = model.start_chat(
            history=request.conversationHistory
        )
        
        # Envía el nuevo mensaje del usuario a la sesión de chat activa.
        response = chat_session.send_message(request.prompt)
        
        # Devuelve la respuesta de texto generada por la IA.
        return {"text": response.text}
        
    except Exception as e:
        print(f"Error during Gemini content generation: {e}")
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred while generating content: {str(e)}")

@app.get("/")
def read_root():
    return {"status": "FDG Constructions AI Backend v4.0 is live and running."}