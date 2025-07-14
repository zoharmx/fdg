# main.py
import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import google.generativeai as genai
from dotenv import load_dotenv

# Carga la variable de entorno para desarrollo local y producción
load_dotenv() 

# Configura la app de FastAPI
app = FastAPI()

# Configura CORS para permitir solicitudes desde tu frontend de producción
# ¡IMPORTANTE! Reemplaza "*" con tu dominio de producción real.
# Si necesitas permitir acceso desde localhost para desarrollo, puedes añadirlo también.
PRODUCTION_DOMAIN = "https://fdgconstructions.site"
ALLOW_ORIGINS = [
    PRODUCTION_DOMAIN,
    "http://localhost:8000", # Permite localhost para desarrollo si es necesario
    "http://127.0.0.1:8000", # Permite localhost para desarrollo si es necesario
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOW_ORIGINS, 
    allow_credentials=True,
    allow_methods=["*"], # Permite todos los métodos HTTP (GET, POST, etc.)
    allow_headers=["*"], # Permite todas las cabeceras
)

# Configura la clave de API de Gemini de forma segura
# Asegúrate de que la variable de entorno GEMINI_API_KEY esté configurada en Render
try:
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    if not gemini_api_key:
        raise ValueError("GEMINI_API_KEY environment variable not set.")
    genai.configure(api_key=gemini_api_key)
except ValueError as ve:
    print(f"Configuration Error: {ve}")
    # En un entorno de producción, podrías querer registrar esto o lanzar una excepción más fuerte
except Exception as e:
    print(f"An unexpected error occurred during Gemini API configuration: {e}")

# Define el modelo de datos para la solicitud del prompt
class PromptRequest(BaseModel):
    prompt: str

@app.post("/api/generate")
async def generate_content_route(request: PromptRequest):
    """
    Endpoint para recibir un prompt del frontend y devolver la respuesta de Gemini.
    """
    if not request.prompt:
        raise HTTPException(status_code=400, detail="No prompt provided.")

    try:
        # Usamos un modelo más capaz como 'gemini-1.5-pro-latest' o 'gemini-1.5-flash-latest'
        # gemini-1.5-pro-latest es más potente, gemini-1.5-flash-latest es más rápido y económico
        model = genai.GenerativeModel('gemini-1.5-pro-latest') 
        
        # Genera contenido usando el prompt
        response = model.generate_content(request.prompt)
        
        # Devuelve el texto de la respuesta
        return {"text": response.text}
        
    except Exception as e:
        print(f"Error in /api/generate: {str(e)}") # Log the error for debugging
        raise HTTPException(status_code=500, detail=f"Failed to generate content: {str(e)}")

@app.get("/")
def read_root():
    """Endpoint raíz para verificar que el backend está funcionando."""
    return {"status": "FDG Constructions Backend API is running"}