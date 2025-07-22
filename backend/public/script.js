// script.js
const clasificacionesPredefinidas = [
  { label: "hoja saludable", mensaje: "🌱 La hoja parece saludable. No hay signos de plaga." },
  { label: "hoja enferma", mensaje: "⚠️ La hoja parece enferma. Revisa por plagas o deficiencias." },
  { label: "hoja seca", mensaje: "💧 La hoja está seca. Considera aumentar el riego." }
];

const baseURL = window.location.origin; // para saber la ip de la pagina

function mostrarImagen(event) {
  const vista = document.getElementById("vistaPrevia");
  if (event.target.files && event.target.files[0]) {
    vista.src = URL.createObjectURL(event.target.files[0]);
    vista.classList.remove("hidden");
  } else {
    vista.classList.add("hidden");
  }
}

async function clasificarHoja() {
  const resultado = document.getElementById("resultado");
  // Simular que ya se ejecutó enviarYOLO y obtener las detecciones
  const detecciones = document.getElementById("detecciones").textContent;
  if (detecciones) {
    const parsedDetections = JSON.parse(detecciones);
    if (parsedDetections.length > 0) {
      // Si hay detecciones, asumir que son plagas (aunque sea genérico por ahora)
      resultado.innerText = "⚠️ La hoja parece enferma. Revisa por plagas o deficiencias.";
    } else {
      resultado.innerText = "🌱 La hoja parece saludable. No hay signos de plaga.";
    }
  } else {
    resultado.innerText = "🌱 La hoja parece saludable. No hay signos de plaga.";
  }
}

function predecirSalud() {
  const resultado = document.getElementById("resultado");
  const temp = parseFloat(document.getElementById("temp").textContent);
  const hum = parseFloat(document.getElementById("hum").textContent);
  const detecciones = document.getElementById("detecciones").textContent;

  let mensaje = "🌿 El cultivo está en buen estado.";
  if (temp < 20 || temp > 35) {
    mensaje = "⚠️ La temperatura está fuera del rango óptimo (20-35°C).";
  } else if (hum < 40) {
    mensaje = "⚠️ La humedad es baja (<40%). Considera regar.";
  } else if (detecciones && JSON.parse(detecciones).length > 0) {
    mensaje = "⚠️ Posibles signos de plagas detectados. Revisa el cultivo.";
  }
  resultado.innerText = mensaje;
}

function recomendarAccion() {
  const resultado = document.getElementById("resultado");
  const temp = parseFloat(document.getElementById("temp").textContent);
  const hum = parseFloat(document.getElementById("hum").textContent);
  const detecciones = document.getElementById("detecciones").textContent;

  let mensaje = "✅ No se requiere acción inmediata.";
  if (hum < 40) {
    mensaje = "💧 Se recomienda regar hoy debido a baja humedad.";
  } else if (temp > 35) {
    mensaje = "🌡️ Se recomienda ventilación o sombra por alta temperatura.";
  } else if (detecciones && JSON.parse(detecciones).length > 0) {
    mensaje = "⚠️ Revisa el cultivo por posibles plagas detectadas.";
  }
  resultado.innerText = mensaje;
}

function hablarConAsistente() {
  const resultado = document.getElementById("resultado");
  const temp = parseFloat(document.getElementById("temp").textContent);
  const hum = parseFloat(document.getElementById("hum").textContent);
  const detecciones = document.getElementById("detecciones").textContent;

  let mensaje = "Hola agricultor, tus cultivos están en buen estado.";
  if (hum < 40) {
    mensaje = "Hola agricultor, considera regar hoy por baja humedad.";
  } else if (temp > 35) {
    mensaje = "Hola agricultor, ventila o da sombra por alta temperatura.";
  } else if (detecciones && JSON.parse(detecciones).length > 0) {
    mensaje = "Hola agricultor, revisa tus cultivos por posibles plagas.";
  }

  resultado.innerText = "🗣 " + mensaje;
  const speech = new SpeechSynthesisUtterance(mensaje);
  speech.lang = "es-ES";
  window.speechSynthesis.speak(speech);
}

async function enviarYOLO() {
  const input = document.getElementById("inputImagen");
  const modelo = document.getElementById("modeloVision").value;
  const resultado = document.getElementById("imagenResultado");
  const detecciones = document.getElementById("detecciones");

  if (!input.files[0]) {
    alert("Por favor selecciona una imagen.");
    return;
  }

  const formData = new FormData();
  formData.append("image", input.files[0]);
  formData.append("model", modelo);

  try {
    const res = await fetch(`${baseURL}/predict`, {
      method: "POST",
      body: formData
    });

    console.log("Estado de la respuesta:", res.status); // Log para depuración
    const data = await res.json();
    console.log("Datos recibidos:", data); // Log para depuración

    if (data.image && data.detections) {
      resultado.src = `data:image/jpeg;base64,${data.image}`;
      resultado.classList.remove("hidden");
      detecciones.textContent = JSON.stringify(data.detections, null, 2);
    } else {
      detecciones.textContent = "❌ Respuesta del servidor incompleta.";
    }
  } catch (err) {
    console.error("Error al enviar la imagen:", err);
    detecciones.textContent = "❌ Error al procesar la imagen.";
  }
}

async function enviarPregunta() {
  const input = document.getElementById("userInput");
  const chat = document.getElementById("chatBox");
  const pregunta = input.value.trim();
  if (pregunta === "") return;

  chat.innerHTML += `<div><strong>Tú:</strong> ${pregunta}</div>`;

  try {
    const respuesta = await fetch(`${baseURL}/chat`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ pregunta: pregunta }),
    });

    console.log("Estado de la respuesta:", respuesta.status); // Log para depuración
    const data = await respuesta.json();
    console.log("Datos recibidos:", data); // Log para depuración

    const output = data.respuesta || "[sin respuesta]";
    chat.innerHTML += `<div><strong>Asistente:</strong> ${output}</div>`;
  } catch (err) {
    console.error("Error al obtener respuesta:", err);
    chat.innerHTML += `<div><strong>Asistente:</strong> Error al obtener respuesta del servidor.</div>`;
  }

  chat.scrollTop = chat.scrollHeight;
  input.value = "";
}