import os
from dotenv import load_dotenv
from typing import Dict
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableBranch, RunnableLambda, RunnablePassthrough
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# --- AGREGA ESTO PARA REPARAR EL ERROR ---
if "SSL_CERT_FILE" in os.environ:
    del os.environ["SSL_CERT_FILE"]
# -----------------------------------------

# 1. Configuración del Modelo
# Asegúrate de tener instalada la librería: pip install langchain-openai
# os.environ["OPENAI_API_KEY"] = "tu_api_key_aqui"

load_dotenv()# Cargamo las variables al sistema (nuestro OPENAI_API_KEY)

model = ChatOpenAI(
    model="gpt-4o", # O "gpt-3.5-turbo"
    temperature=0.2
)

# 2. Definición de los Prompts
prompt_urgencias = PromptTemplate.from_template(
    "SISTEMA DE EMERGENCIAS: El paciente presenta síntomas críticos: {input}. "
    "Genera un protocolo de estabilización inmediata y alerta a triaje nivel 1."
)

prompt_pediatria = PromptTemplate.from_template(
    "CONSULTA PEDIÁTRICA: El paciente es un menor con los síntomas: {input}. "
    "Analiza dosis por peso y enfoque en desarrollo infantil."
)

prompt_general = PromptTemplate.from_template(
    "MEDICINA GENERAL: Analiza los siguientes síntomas: {input}. "
    "Provee un diagnóstico diferencial de rutina."
)

# 3. Lógica de Clasificación
def classify_case(data: Dict) -> str:
    # Si 'data' es un dict con 'input', extraemos el texto
    text = data["input"].lower() if isinstance(data, dict) else str(data).lower()
    
    if any(word in text for word in ["pecho", "infarto", "respirar", "grave"]):
        return "urgencias"
    elif any(word in text for word in ["niño", "bebe", "hijo", "años"]):
        return "pediatria"
    else:
        return "general"

# 4. Construcción del RunnableBranch
branch = RunnableBranch(
    (lambda x: x["topic"] == "urgencias", prompt_urgencias),
    (lambda x: x["topic"] == "pediatria", prompt_pediatria),
    prompt_general
)

# 5. La Secuencia Maestra
chain = (
    {
        "topic": RunnableLambda(classify_case), 
        "input": lambda x: x["input"] 
    }
    | branch 
    | model 
    | StrOutputParser()
)

# 6. Ejecución
resultado = chain.invoke({"input": "Mi hijo tiene mucha fiebre y tos"})
print(resultado)