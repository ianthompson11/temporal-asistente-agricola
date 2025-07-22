const express = require("express");
const axios = require("axios");
const path = require("path");

const app = express();
const PORT = 3000;

// Middleware
app.use(express.json());
app.use(express.static(path.join(__dirname, "public")));

// 👉 Si alguien accede a "/" devuelve index.html automáticamente
app.get("/", (req, res) => {
  res.sendFile(path.join(__dirname, "public", "index.html"));
});

app.post("/chat", async (req, res) => {
  const pregunta = req.body.pregunta;

  try {
    const respuesta = await axios.post(
      "http://localhost:5000/chat", // Nueva API local
      {
        pregunta: pregunta,
      },
      {
        headers: {
          "Content-Type": "application/json",
        },
      }
    );

    const output = respuesta.data.respuesta || "[sin respuesta]";
    res.json({ respuesta: output });
  } catch (error) {
    console.error("Error al conectar con el modelo local:", error.message);
    res.status(500).json({ respuesta: "Error al obtener respuesta del modelo." });
  }
});

app.listen(PORT, () => {
  console.log(`Servidor escuchando en http://localhost:${PORT}`);
});