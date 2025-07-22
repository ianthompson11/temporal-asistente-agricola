from unsloth import FastLanguageModel
from flask import Flask, request, jsonify, send_from_directory, render_template_string
from flask_cors import CORS
from transformers import pipeline
import torch
import logging
import requests
import base64
import json
import os
import tempfile
import uuid
import numpy as np
from PIL import Image
from ultralytics import YOLO
from pyngrok import ngrok
import io

# ---------------- CONFIGURACIÓN DE MODELO LLM ----------------
max_seq_length = 512
dtype = torch.float16
load_in_4bit = True
model_name = "Julio1023/phi2-merged"

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_name,
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Agrega una lista para almacenar el historial de la conversación
conversation_history = []

def generar_respuesta(prompt, max_new_tokens=300):
    global conversation_history
    
    # Añade la nueva pregunta al historial
    conversation_history.append({"role": "user", "content": prompt})
    
    # Construye el prompt con el historial
    formatted_prompt = "Instrucciones: Responde únicamente con la información solicitada, manteniendo el contexto de la conversación anterior.\n\n"
    for message in conversation_history:
        if message["role"] == "user":
            formatted_prompt += f"Pregunta: {message['content']}\n"
        else:
            formatted_prompt += f"Respuesta: {message['content']}\n"
    formatted_prompt += "Respuesta:"
    
    # Tokeniza y genera la respuesta
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(device)
    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.5,
        top_p=0.85,
        eos_token_id=tokenizer.eos_token_id,
    )
    respuesta = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    
    # Limpia la respuesta
    respuesta = respuesta.replace(formatted_prompt, "").strip()
    
    # Añade la respuesta al historial
    conversation_history.append({"role": "assistant", "content": respuesta})
    
    # Opcional: Limita el historial para no exceder max_seq_length
    if len(tokenizer.encode(formatted_prompt)) > max_seq_length * 0.8:
        conversation_history.pop(0)  # Elimina el mensaje más antiguo si es necesario
    
    return respuesta

# ---------------- CONFIGURACIÓN GENERAL ----------------
ESP32_IP = os.environ.get("ESP32_IP", "http://192.168.1.139")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__, static_folder='public')
CORS(app)

# ---------------- CARGA DE MODELOS YOLO ----------------
try:
    logger.info("Cargando modelos YOLO...")
    models = {
        "deteccion_plagas": YOLO("/content/temporal-asistente-agricola/backend/deteccion_plagas.pt"),
        "deteccion_enfermedades": YOLO("/content/temporal-asistente-agricola/backend/deteccion_enfermedades.pt"),
        "plant_protect": YOLO("/content/temporal-asistente-agricola/backend/plant_protect.pt")
    }
    logger.info("Modelos YOLO cargados exitosamente.")
except Exception as e:
    logger.error(f"Error cargando modelos YOLO: {e}")
    models = {}

# ---------------- RUTAS ----------------

@app.route("/")
def index():
    return send_from_directory(app.static_folder, 'index.html')

@app.route("/<path:path>")
def serve_static(path):
    return send_from_directory(app.static_folder, path)

@app.route("/api/sensores")
def api_sensores():
    try:
        respuesta = requests.get(f"{ESP32_IP}/datos", timeout=3)
        return respuesta.json()
    except Exception as e:
        return {"error": "No se pudo conectar al ESP32"}, 500

@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.get_json()
        pregunta = data.get("pregunta", "")
        if not pregunta:
            return jsonify({"respuesta": "No se proporcionó una pregunta."}), 400
        logger.info(f"Pregunta recibida: {pregunta}")
        respuesta = generar_respuesta(pregunta)
        return jsonify({"respuesta": respuesta})
    except Exception as e:
        logger.error(f"Error en /chat: {e}")
        return jsonify({"respuesta": "Error al procesar la pregunta."}), 500

@app.route("/predict", methods=["POST"])
def predict():
    try:
        model_name = request.form.get("model")
        file = request.files.get("image")

        if not model_name or model_name not in models:
            return jsonify({"error": "Modelo YOLO no válido."}), 400
        if not file:
            return jsonify({"error": "Imagen no proporcionada."}), 400

        if model_name == "plant_protect":
            temp_dir = tempfile.gettempdir()
            temp_filename = f"{uuid.uuid4()}.jpg"
            temp_image_path = os.path.join(temp_dir, temp_filename)
            file.save(temp_image_path)

            results = models[model_name].predict(
                source=temp_image_path,
                conf=0.3,
                save=True,
                project=temp_dir,
                name='deteccion_hoja',
                exist_ok=True
            )

            saved_path = os.path.join(temp_dir, 'deteccion_hoja', os.path.basename(temp_image_path))
            with open(saved_path, "rb") as f:
                encoded_img = base64.b64encode(f.read()).decode()

            detections = json.loads(results[0].tojson()) if results[0].boxes else []

        else:
            img_bytes = file.read()
            image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            img_np = np.array(image)

            results = models[model_name].predict(image)
            annotated_img = results[0].plot(img=img_np)

            buffered = io.BytesIO()
            Image.fromarray(annotated_img).save(buffered, format="JPEG")
            encoded_img = base64.b64encode(buffered.getvalue()).decode()

            detections = json.loads(results[0].tojson()) if results[0].boxes else []

        # ---- Generar descripción textual basada en las detecciones ----
        if detections:
            # Extraer nombres, corregir y eliminar duplicados preservando orden
            nombres_raw = [d.get("name", "objeto desconocido").lower() for d in detections]

            # Tabla de correcciones (puedes ampliarla según tus etiquetas reales)
            correcciones = {
                "arana roja": "araña roja",
                "mosca blanca": "mosca blanca",
                "roya": "roya",
                # agrega más si tu modelo tiene etiquetas con errores
            }

            nombres_corregidos = [correcciones.get(n, n) for n in nombres_raw]

            # Eliminar duplicados preservando orden
            nombres_unicos = list(dict.fromkeys(nombres_corregidos))

            # Construir descripción final
            descripcion_texto = "¿Qué es " + " y ".join(nombres_unicos) + "?"
        else:
            descripcion_texto = "No se detectaron objetos."

        # ---- Enviar resultado completo al frontend ----
        return jsonify({
            "image": encoded_img,
            "detections": detections,
            "descripcion_texto": descripcion_texto
        })

    except Exception as e:
        logger.error(f"Error en /predict: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/predict_stream", methods=["POST"])
def predict_stream():
    try:
        model_name = request.form.get("model")
        if model_name != "plant_protect":
            return jsonify({"error": "Solo plant_protect es compatible con streaming."}), 400

        image_data = request.form.get("image")
        if not image_data:
            return jsonify({"error": "No se proporcionó una imagen."}), 400

        image_data = image_data.split(",")[1]
        img_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(img_bytes)).convert("RGB")

        temp_dir = tempfile.gettempdir()
        temp_filename = f"{uuid.uuid4()}.jpg"
        temp_image_path = os.path.join(temp_dir, temp_filename)
        image.save(temp_image_path)

        results = models[model_name].predict(
            source=temp_image_path,
            conf=0.25,
            save=True,
            project=temp_dir,
            name='deteccion_hoja',
            exist_ok=True
        )

        saved_path = os.path.join(temp_dir, 'deteccion_hoja', os.path.basename(temp_image_path))
        with open(saved_path, "rb") as f:
            encoded_img = base64.b64encode(f.read()).decode()

        detections = json.loads(results[0].tojson()) if results[0].boxes else []

        return jsonify({"image": encoded_img, "detections": detections})
    except Exception as e:
        logger.error(f"Error en /predict_stream: {e}")
        return jsonify({"error": str(e)}), 500

# ---------------- INICIAR NGROK + FLASK ----------------
if __name__ == "__main__":
    port = 8000
    public_url = ngrok.connect(port)
    print(f" * Servidor disponible en: {public_url}")
    app.run(host="0.0.0.0", port=port, debug=True, use_reloader=False)
