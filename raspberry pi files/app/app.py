from flask import Flask, render_template, Response
from picamera2 import Picamera2
import cv2
import numpy as np
import onnxruntime as ort

app = Flask(__name__)

# Inicializar cámara
picam2 = Picamera2()
picam2.start()

# Cargar modelo ONNX (640x640 fijo)
session = ort.InferenceSession(
    "[path to your model]/best.onnx",
    providers=["CPUExecutionProvider"]
)
input_name = session.get_inputs()[0].name
target_size = (640, 640)

# --- Preprocesamiento con letterbox ---
def letterbox(img, new_shape=(640, 640), color=(114, 114, 114)):
    shape = img.shape[:2]  # alto, ancho
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
    dw = (new_shape[1] - new_unpad[0]) / 2
    dh = (new_shape[0] - new_unpad[1]) / 2

    img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right,
                             cv2.BORDER_CONSTANT, value=color)
    return img, r, dw, dh, new_unpad

def preprocess(frame):
    if frame.shape[2] == 4:
        frame = frame[:, :, :3]
    img, r, dw, dh, new_unpad = letterbox(frame, target_size)
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)
    return img, r, dw, dh, new_unpad

# --- NMS ---
def nms(boxes, scores, iou_threshold=0.5):
    idxs = np.argsort(scores)[::-1]
    keep = []
    while len(idxs) > 0:
        i = idxs[0]
        keep.append(i)
        if len(idxs) == 1:
            break
        ious = compute_iou(boxes[i], boxes[idxs[1:]])
        idxs = idxs[1:][ious < iou_threshold]
    return keep

def compute_iou(box1, boxes):
    x1 = np.maximum(box1[0], boxes[:,0])
    y1 = np.maximum(box1[1], boxes[:,1])
    x2 = np.minimum(box1[2], boxes[:,2])
    y2 = np.minimum(box1[3], boxes[:,3])
    inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    area1 = (box1[2]-box1[0]) * (box1[3]-box1[1])
    area2 = (boxes[:,2]-boxes[:,0]) * (boxes[:,3]-boxes[:,1])
    union = area1 + area2 - inter
    return inter / (union + 1e-6)

# --- Ajuste de cajas ---
def scale_boxes(boxes, r, dw, dh, original_shape):
    # Restar padding y dividir por el factor de escala
    boxes[:, 0] = (boxes[:, 0] - dw) / r
    boxes[:, 1] = (boxes[:, 1] - dh) / r
    boxes[:, 2] = (boxes[:, 2] - dw) / r
    boxes[:, 3] = (boxes[:, 3] - dh) / r

    # Limitar dentro de la imagen original
    h, w = original_shape
    boxes[:, 0] = np.clip(boxes[:, 0], 0, w)
    boxes[:, 2] = np.clip(boxes[:, 2], 0, w)
    boxes[:, 1] = np.clip(boxes[:, 1], 0, h)
    boxes[:, 3] = np.clip(boxes[:, 3], 0, h)
    return boxes

def postprocess(frame, outputs, r, dw, dh, new_unpad):
    preds = outputs[0][0]  # batch 0
    # Convertir de [cx, cy, w, h] a [x1, y1, x2, y2]
    cx, cy, w_box, h_box = preds[:,0], preds[:,1], preds[:,2], preds[:,3]
    x1 = cx - w_box/2
    y1 = cy - h_box/2
    x2 = cx + w_box/2
    y2 = cy + h_box/2
    boxes = np.stack([x1, y1, x2, y2], axis=1)

    scores = preds[:, 4]
    classes = np.argmax(preds[:, 5:], axis=1)

    mask = scores > 0.5
    boxes, scores, classes = boxes[mask], scores[mask], classes[mask]

    if len(boxes) == 0:
        return frame

    # Dibujar área de detección fijo
    pad_x, pad_y = int(dw), int(dh)
    cv2.rectangle(frame,
                  (pad_x, pad_y),
                  (pad_x + new_unpad[0], pad_y + new_unpad[1]),
                  (255,0,0), 2)
    cv2.putText(frame, "Area de deteccion", (pad_x, pad_y-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)

    # Escalar cajas al espacio original
    boxes = scale_boxes(boxes, r, dw, dh, frame.shape[:2])

    keep = nms(boxes, scores)
    boxes, scores, classes = boxes[keep], scores[keep], classes[keep]

    for (x1, y1, x2, y2), score, cls in zip(boxes, scores, classes):
        cv2.rectangle(frame, (int(x1), int(y1)),
                      (int(x2), int(y2)), (0,255,0), 2)
        cv2.putText(frame, f"Car {score:.2f}",
                    (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0,255,0), 2)

    return frame

def gen_frames():
    while True:
        frame = picam2.capture_array()
        inp, r, dw, dh, new_unpad = preprocess(frame)
        outputs = session.run(None, {input_name: inp})
        frame = postprocess(frame, outputs, r, dw, dh, new_unpad)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)