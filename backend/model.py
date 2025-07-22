from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
import logging
from transformers import pipeline
import torch
from PIL import Image
import io
import base64
from ultralytics import YOLO
import json
import numpy as np
import requests
import os
import tempfile
import uuid

ESP32_IP = os.environ.get("ESP32_IP", "http://192.168.1.139")

# Configurar logging para depuración
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Cargar modelo de lenguaje natural
try:
    logger.info("Cargando el modelo deepset/roberta-base-squad2...")
    pipe = pipeline("question-answering", model="deepset/roberta-base-squad2")
    logger.info("Modelo cargado exitosamente.")
except Exception as e:
    logger.error(f"Error al cargar el modelo: {e}")
    raise

# Función para procesar preguntas
def get_answer(question, context):
    try:
        result = pipe(question=question, context=context)
        logger.info(f"Respuesta generada: {result['answer']}")
        return result['answer']
    except Exception as e:
        logger.error(f"Error al procesar la pregunta: {e}")
        return "[sin respuesta]"

# Cargar modelos YOLO disponibles
try:
    logger.info("Cargando modelos YOLO...")
    models = {
        "deteccion_plagas": YOLO("deteccion_plagas.pt"),
        "deteccion_enfermedades": YOLO("deteccion_enfermedades.pt"),
        "plant_protect": YOLO("plant_protect.pt")
    }
    logger.info("Modelos YOLO cargados exitosamente.")
except Exception as e:
    logger.error(f"Error cargando modelos YOLO: {e}")
    models = {}

# Flask API
app = Flask(__name__, static_folder='public')
CORS(app)

@app.route("/api/sensores")
def api_sensores():
    try:
        respuesta = requests.get(f"{ESP32_IP}/datos", timeout=3)
        return respuesta.json()
    except Exception as e:
        return {"error": "No se pudo conectar al ESP32"}, 500

@app.route('/')
def index():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory(app.static_folder, path)

@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        if not data or 'pregunta' not in data:
            return jsonify({"respuesta": "Error: No se proporcionó una pregunta."}), 400
        pregunta = data['pregunta']
        contexto = "El cultivo de tomate necesita entre 20 y 25 grados de temperatura para crecer óptimamente. Es importante mantener el suelo húmedo y bien drenado. Se recomienda fertilizar con potasio."
        logger.info(f"Recibida pregunta: {pregunta}")
        respuesta = get_answer(pregunta, contexto)
        if respuesta == "[sin respuesta]":
            respuesta = "Lo siento, solo tengo información sobre el cultivo de tomates por ahora."
        return jsonify({"respuesta": respuesta})
    except Exception as e:
        logger.error(f"Error en la ruta /chat: {e}")
        return jsonify({"respuesta": "Error al obtener respuesta del modelo."}), 500

@app.route('/predict', methods=['POST'])
def predict():
    try:
        model_name = request.form.get('model')
        file = request.files.get('image')

        if not model_name or model_name not in models:
            return jsonify({"error": "Modelo YOLO no válido."}), 400
        if not file:
            return jsonify({"error": "Imagen no proporcionada."}), 400

        if model_name == "plant_protect":
            # 1. Guardar imagen temporalmente
            temp_dir = tempfile.gettempdir()
            temp_filename = f"{uuid.uuid4()}.jpg"
            temp_image_path = os.path.join(temp_dir, temp_filename)
            file.save(temp_image_path)

            # 2. Ejecutar predicción
            results = models[model_name].predict(
                source=temp_image_path,
                conf=0.3,
                save=True,
                project=temp_dir,
                name='deteccion_hoja',
                exist_ok=True
            )

            # 3. Leer imagen anotada generada
            saved_path = os.path.join(temp_dir, 'deteccion_hoja', os.path.basename(temp_image_path))
            with open(saved_path, "rb") as f:
                encoded_img = base64.b64encode(f.read()).decode()

            # 4. Obtener detecciones
            detections = json.loads(results[0].tojson()) if results[0].boxes else []

        else:
            # Procesamiento estándar para los demás modelos
            img_bytes = file.read()
            image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            img_np = np.array(image)

            results = models[model_name].predict(image)

            annotated_img = results[0].plot(img=img_np)
            buffered = io.BytesIO()
            Image.fromarray(annotated_img).save(buffered, format="JPEG")
            encoded_img = base64.b64encode(buffered.getvalue()).decode()

            detections = json.loads(results[0].tojson()) if results[0].boxes else []

        return jsonify({
            "image": encoded_img,
            "detections": detections
        })

    except Exception as e:
        logger.error(f"Error en la ruta /predict: {e}")
        return jsonify({"error": f"Error al procesar la imagen: {str(e)}"}), 500

@app.route('/predict_stream', methods=['POST'])
def predict_stream():
    try:
        model_name = request.form.get('model')
        if model_name != "plant_protect":
            return jsonify({"error": "Solo el modelo plant_protect es compatible con streaming."}), 400

        # Obtener la imagen desde el canvas (base64)
        image_data = request.form.get('image')
        if not image_data:
            return jsonify({"error": "No se proporcionó una imagen."}), 400

        # Decodificar la imagen base64
        image_data = image_data.split(',')[1]  # Eliminar el prefijo 'data:image/jpeg;base64,'
        img_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(img_bytes)).convert("RGB")

        # Guardar imagen temporalmente para plant_protect
        temp_dir = tempfile.gettempdir()
        temp_filename = f"{uuid.uuid4()}.jpg"
        temp_image_path = os.path.join(temp_dir, temp_filename)
        image.save(temp_image_path)

        # Ejecutar predicción con YOLO
        results = models[model_name].predict(
            source=temp_image_path,
            conf=0.25,
            save=True,
            project=temp_dir,
            name='deteccion_hoja',
            exist_ok=True
        )

        # Leer imagen anotada generada
        saved_path = os.path.join(temp_dir, 'deteccion_hoja', os.path.basename(temp_image_path))
        with open(saved_path, "rb") as f:
            encoded_img = base64.b64encode(f.read()).decode()

        # Obtener detecciones
        detections = json.loads(results[0].tojson()) if results[0].boxes else []

        return jsonify({
            "image": encoded_img,
            "detections": detections
        })

    except Exception as e:
        logger.error(f"Error en la ruta /predict_stream: {e}")
        return jsonify({"error": f"Error al procesar el frame: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True, ssl_context='adhoc')

