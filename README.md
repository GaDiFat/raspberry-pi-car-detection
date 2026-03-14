# Proyecto de Detección de Autos en Tiempo Real

Este proyecto se desarrolló como parte del programa de Maestría en Sistemas Embebidos en la Universidad Aeronáutica en Querétaro (UNAQ).  
El objetivo es entrenar un modelo YOLOv5 para detección de autos, desplegarlo en una Raspberry Pi y transmitir las detecciones en tiempo real a través de una página web con Flask.

---

## Preparación del Dataset
- Imágenes originales: ./data/training_images/
- Anotaciones: ./data/trainsolutionbounding_boxes.csv
- Resolución utilizada: 380 × 676
- Conversión de bounding boxes a formato YOLO.
- División en train (80%) y val (20%).
- Archivo data.yaml con configuración de clases:
  yaml
  train: dataset/images/train
  val: dataset/images/val

  nc: 1
  names: ['car']
  

---

## Entrenamiento del Modelo
Ejecutar el entrenamiento en tu computadora con:

bash
python yolov5/train.py --img 640 --batch 16 --epochs 50 --data data.yaml --weights yolov5s.pt


---

## Configuración en la Raspberry Pi

1. **Instalar dependencias**
   bash
   sudo apt update && sudo apt install -y \
       python3-numpy \
       python3-opencv \
       python3-flask \
       python3-picamera2 \
       python3-onnxruntime
   

2. **Copiar el modelo entrenado**
   - Transferir el archivo de pesos (best.pt o exportado a ONNX) a la Raspberry Pi.

3. **Ejecutar inferencia con ONNX Runtime**
   - Usar onnxruntime para cargar el modelo y procesar imágenes en tiempo real desde la cámara.

4. **Servidor Flask para streaming**
   - Crear un script Flask que:
     - Capture frames de la cámara con picamera2.
     - Procese cada frame con el modelo YOLO.
     - Dibuje las detecciones en la imagen.
     - Transmita el resultado en una página web accesible desde la red local.

   Ejemplo de ejecución:
   bash
   python app.py
   
   Luego acceder desde el navegador a:
   
   http://<IPdetu_Raspberry>:5000
   

---

## Resultados
- Dataset convertido a formato YOLOv5.
- Modelo entrenado y desplegado en Raspberry Pi.
- Detecciones de autos transmitidas en tiempo real vía Flask.

---

## Créditos
Este proyecto se realizó como parte del programa de Maestría en Ingeniería Aeroespaciasl con acentuación en Sistemas Embebidos en UNAQ, integrando visión por computadora, sistemas embebidos y desarrollo web.
