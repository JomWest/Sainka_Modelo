from flask import Flask, request, jsonify
import cv2
import mediapipe as mp
import joblib
import numpy as np

app = Flask(__name__)

# Cargar el modelo entrenado
model = joblib.load('sign_language_model.pkl')

# Inicializar MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

@app.route('/predict', methods=['POST'])
def predict():
    # Obtener la imagen desde el POST request
    file = request.files['image']
    image = np.frombuffer(file.read(), np.uint8)
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    
    # Guarda la imagen para verificarla
    cv2.imwrite('received_image.jpg', image)

    # Procesar la imagen
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])
            
            landmarks = np.array(landmarks).reshape(1, -1)
            prediction = model.predict(landmarks)[0]
            return jsonify({'prediction': prediction})
    
    return jsonify({'prediction': 'No hand detected'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
