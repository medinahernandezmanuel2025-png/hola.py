import os
import time
import speech_recognition as sr
import pyttsx3
from google import genai
from google.genai import types

# 1. Configuración de la Voz (pyttsx3)
engine = pyttsx3.init()
voices = engine.getProperty('voices')
# Intentar buscar una voz masculina o en español (puedes cambiar el índice 0, 1, 2 según tu PC)
engine.setProperty('voice', voices[0].id) 
engine.setProperty('rate', 180) # Velocidad del habla
def hablar(texto):
    
    """Hace que la IA hable."""
    print(f"JARVIS: {texto}")
    engine.say(texto)
    engine.runAndWait()  # Esto es suficiente para procesar el habla de forma síncrona
    # SE ELIMINÓ: engine.stop() para evitar que el motor se destruya

    

# 2. Configuración del "Cerebro" (Google Gemini)
# Reemplaza 'TU_API_KEY_AQUÍ' con tu clave real de Google AI Studio
# O mejor aún, configúrala en tu sistema como variable de entorno GEMINI_API_KEY
os.environ["GEMINI_API_KEY"] = 

try:
    client = genai.Client()
except Exception as e:
    print("Error al conectar con la API de Gemini. Verifica tu API Key.")
    client = None

# Configurar la personalidad de JARVIS mediante el System Instruction
config_jarvis = types.GenerateContentConfig(
    system_instruction=(
        "Eres JARVIS, el asistente de inteligencia artificial inteligente, sofisticado "
        "y leal de Tony Stark. Responde siempre en español, de manera educada pero con un toque "
        "de humor sutil y sarcasmo seco británico. Mantén tus respuestas cortas, directas y eficientes."
    ),
    temperature=0.7
)

# 3. Configuración del "Oído" (Speech Recognition)
recognizer = sr.Recognizer()

def escuchar():
    """Captura el audio del micrófono y lo convierte a texto."""
    with sr.Microphone() as source:
        print("\n[Escuchando...] Di algo...")
        # Ajustar el ruido ambiental brevemente
        recognizer.adjust_for_ambient_noise(source, duration=0.5)
        try:
            audio = recognizer.listen(source, timeout=5, phrase_time_limit=10)
            print("[Procesando voz...]")
            # Usamos el reconocimiento de Google nativo de la librería para pasar de audio a texto
            query = recognizer.recognize_google(audio, language="es-MX")
            print(f"Tú dijiste: {query}")
            return query.lower()
        except sr.WaitTimeoutError:
            return ""
        except sr.UnknownValueError:
            print("JARVIS: Disculpe, señor, no logré entender el audio.")
            return ""
        except sr.RequestError:
            print("JARVIS: Hay un problema de conexión con el servicio de reconocimiento.")
            return ""

# 4. Bucle Principal de Ejecución
if __name__ == "__main__":
    hablar("Sistemas en línea, señor. ¿En qué puedo ayudarle hoy?")
    
    while True:
        pedido = escuchar()
        
        if not pedido:
            continue
            
        # Comandos de apagado
        if "adiós" in pedido or "apágate" in pedido or "descansa" in pedido:
            hablar("Entendido, señor. Desconectando sistemas. Que tenga un buen día.")
            break
            
        # Ejecución de acciones locales (Ejemplo: abrir el navegador)
        if "abre youtube" in pedido:
            hablar("Abriendo YouTube de inmediato, señor.")
            time.sleep(3)
            import webbrowser
            webbrowser.open("https://youtube.com")
            continue
        
        if "canción" in pedido:
           hablar("Abriendo YouTube de inmediato, señor.")
           import webbrowser
           webbrowser.open("https://www.youtube.com/watch?v=brrkTawr0n4&list=RDbrrkTawr0n4&start_radio=1")
           continue
                    
            
        if "abre google" in pedido:
            hablar("Abriendo el buscador, señor.")
            import webbrowser
            webbrowser.open("https://google.com")
            continue

        if "abre spotify" in pedido:
                    hablar("Abriendo el buscador, señor.")
                    import webbrowser
                    webbrowser.open("https://open.spotify.com/")
                    continue
        # Si no es un comando local, procesar la respuesta con Inteligencia Artificial
        if client:
            try:
                # Usamos gemini-2.5-flash que es rápido e ideal para asistentes
                response = client.models.generate_content(
                    model="gemini-3.1-flash-lite",
                    contents=pedido,
                    config=config_jarvis
                )
                hablar(response.text)
            except Exception as e:
                hablar("Lo lamento, señor. He experimentado un percance al procesar su solicitud en mi núcleo de IA.")
                print(f"Error de API: {e}")
        else:
            hablar("Lo siento, señor. Mi cerebro de IA no está conectado porque falta la clave de API.")
