import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import google.generativeai as genai
from dotenv import load_dotenv

# Carga la variable de entorno para desarrollo local (no afecta a producción)
load_dotenv()

# Configura la app de FastAPI
app = FastAPI()

# --- CAMBIO IMPORTANTE PARA PRODUCCIÓN ---
# Ahora solo permitimos solicitudes desde tu dominio oficial.
origins = [
    "https://fdgconstructions.site",
    "http://fdgconstructions.site",
    "https://www.fdgconstructions.site",
    "http://www.fdgconstructions.site",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins, # Usamos la lista de orígenes permitidos
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configura la clave de API de Gemini de forma segura
try:
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
except Exception as e:
    print(f"Error configuring Gemini API: {e}")

# Define el modelo de datos para la solicitud
class PromptRequest(BaseModel):
    prompt: str

@app.post("/api/generate")
async def generate_content_route(request: PromptRequest):
    if not request.prompt:
        raise HTTPException(status_code=400, detail="No prompt provided.")

    try:
        # Usamos el modelo Pro para la mejor calidad de conversación
        model = genai.GenerativeModel('gemini-1.5-pro') 
        response = model.generate_content(request.prompt)
        return {"text": response.text}
    except Exception as e:
        # Devuelve un error más detallado para facilitar la depuración si algo falla
        raise HTTPException(status_code=500, detail=f"Error generating content from Gemini: {str(e)}")

@app.get("/")
def read_root():
    return {"status": "FDG Constructions AI Backend is running"}
