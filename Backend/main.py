# main.py
import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import google.generativeai as genai
from dotenv import load_dotenv

# Carga la variable de entorno para desarrollo local
load_dotenv()

# Configura la app de FastAPI
app = FastAPI()

# Configura CORS para permitir solicitudes desde tu frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Para producción, reemplaza "*" con el dominio de tu sitio de Hostinger
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configura la clave de API de Gemini de forma segura
try:
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
except Exception as e:
    print(f"Error al configurar la API de Gemini: {e}")

# Define el modelo de datos para la solicitud
class PromptRequest(BaseModel):
    prompt: str

@app.post("/api/generate")
async def generate_content_route(request: PromptRequest):
    if not request.prompt:
        raise HTTPException(status_code=400, detail="No se proporcionó un prompt.")

    try:
        model = genai.GenerativeModel('gemini-1.5-pro') # Usamos el modelo PRO para el chat
        response = model.generate_content(request.prompt)
        return {"text": response.text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al generar contenido: {str(e)}")

@app.get("/")
def read_root():
    return {"status": "Backend de FDG Constructions está funcionando"}