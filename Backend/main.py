import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import google.generativeai as genai
from dotenv import load_dotenv

# Carga la variable de entorno para desarrollo local (no afectará a Render)
load_dotenv()

# Configura la app de FastAPI
app = FastAPI(
    title="FDG Constructions AI Backend",
    description="API to connect the conversational frontend with Google's Gemini AI.",
    version="1.0.0"
)

# --- Configuración de CORS para Producción ---
# Se especifica el dominio exacto del frontend para máxima seguridad.
# Render maneja la comunicación interna, pero esto previene que otros sitios usen tu API.
origins = [
    "https://fdgconstructions.site",
    # Si tienes un dominio de vista previa en Render, agrégalo aquí también.
    # "https://your-render-preview-domain.onrender.com",
    # Para pruebas locales, puedes agregar:
    # "http://127.0.0.1:5500",
    # "http://localhost:8000"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins, # <-- ¡CAMBIO IMPORTANTE!
    allow_credentials=True,
    allow_methods=["POST", "GET"], # Solo permitir los métodos necesarios
    allow_headers=["Content-Type"],   # Solo permitir los encabezados necesarios
)

# Configura la clave de API de Gemini de forma segura desde las variables de entorno de Render
try:
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    if not gemini_api_key:
        raise ValueError("GEMINI_API_KEY environment variable not found.")
    genai.configure(api_key=gemini_api_key)
except Exception as e:
    # Esto se imprimirá en los logs de Render si hay un problema al iniciar.
    print(f"CRITICAL ERROR: Could not configure Gemini API. {e}")

# Define el modelo de datos para la solicitud que llega del frontend
class PromptRequest(BaseModel):
    prompt: str
    conversationHistory: list # Incluimos el historial para dar contexto a la IA

@app.post("/api/generate")
async def generate_content_route(request: PromptRequest):
    if not request.prompt:
        raise HTTPException(status_code=400, detail="No prompt provided.")

    try:
        # Usamos un modelo pro para un chat más sofisticado
        model = genai.GenerativeModel('gemini-1.5-pro')

        # El historial de la conversación se pasa directamente al modelo
        # para que cada nueva respuesta tenga el contexto completo.
        response = model.generate_content(
            request.conversationHistory + [{'role': 'user', 'parts': [request.prompt]}]
        )

        return {"text": response.text}
    except Exception as e:
        # Log del error para depuración en Render
        print(f"Error during Gemini content generation: {e}")
        raise HTTPException(status_code=500, detail=f"Error generating content: {str(e)}")

@app.get("/")
def read_root():
    return {"status": "FDG Constructions AI Backend is live and running."}