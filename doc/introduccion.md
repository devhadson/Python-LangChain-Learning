La **temperature** (temperatura) en LangChain **es un parámetro hiperparámetro que se utiliza para configurar la aleatoriedad, creatividad y predictibilidad de las respuestas generadas por un Modelo de Lenguaje Grande** (LLM).

En términos técnicos, controla la probabilidad de que el modelo seleccione palabras o frases menos probables durante la generación de texto. 

## Características clave:

### 1. ¿Cómo funciona?
- **Temperatura Baja (cercana a 0)**: El modelo es más conservador, determinista y lógico. Elige casi siempre la palabra siguiente más probable.
    - _Uso_: Resúmenes, preguntas y respuestas precisas, extracción de datos.
- **Temperatura Alta (cercana a 1 o superior)**: El modelo se vuelve más "creativo", aleatorio y diverso. Elige palabras menos probables, lo que puede aumentar la variedad del texto pero también el riesgo de errores (alucinaciones).
    - _Uso_: Escritura creativa, lluvia de ideas, generación de historias. 

### 2. Configuración en LangChain
En LangChain, la temperatura se configura al inicializar el modelo (por ejemplo, con OpenAI o Ollama):

```python
from langchain_openai import ChatOpenAI

# Temperatura baja para respuestas precisas
llm_preciso = ChatOpenAI(temperature=0)

# Temperatura alta para escritura creativa
llm_creativo = ChatOpenAI(temperature=0.8)
```

### 3. Valores típicos
- **0.0 - 0.3**: Muy determinista, alta consistencia.
- **0.7**: Valor por defecto común, equilibrado.
- **0.8 - 1.0+**: Alta creatividad, mayor riesgo de incoherencia

### Resumen
La temperatura te permite **equilibrar la consistencia frente a la creatividad** en tus aplicaciones desarrolladas con LangChain.