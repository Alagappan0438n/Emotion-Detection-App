from flask import Flask, request, jsonify, render_template
import cv2
import numpy as np
from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing.image import img_to_array
from werkzeug.utils import secure_filename
import os
import base64

app = Flask(__name__)

# Load the model and weights
model = model_from_json(open("model.json", "r").read())
model.load_weights('model.h5')

# Load face detection model
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Emotion labels
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Function to detect and classify emotion from image
def detect_emotion_from_image(img):
    gray_frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

    emotions = []
    for (x, y, w, h) in faces:
        face_roi = gray_frame[y:y + h, x:x + w]
        face_roi_resized = cv2.resize(face_roi, (48, 48))

        img_array = img_to_array(face_roi_resized)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0

        predictions = model.predict(img_array)
        max_index = np.argmax(predictions[0])
        predicted_emotion = emotion_labels[max_index]
        confidence = np.max(predictions[0]) * 100
        emotions.append({'emotion': predicted_emotion, 'confidence': confidence})

    return emotions

# Function to process video and detect emotions
def detect_emotion_from_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_emotions = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

        frame_result = []  # To hold emotions for this frame

        for (x, y, w, h) in faces:
            face_roi = gray_frame[y:y + h, x:x + w]
            face_roi_resized = cv2.resize(face_roi, (48, 48))

            img_array = img_to_array(face_roi_resized)
            img_array = np.expand_dims(img_array, axis=0)
            img_array /= 255.0

            predictions = model.predict(img_array)
            max_index = np.argmax(predictions[0])
            predicted_emotion = emotion_labels[max_index]
            confidence = np.max(predictions[0]) * 100
            
            # Store the detected emotion with confidence for this frame
            frame_result.append({'emotion': predicted_emotion, 'confidence': confidence})

        frame_emotions.append(frame_result)

    cap.release()
    return frame_emotions

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload_image')
def upload_image_page():
    return render_template('upload_image.html')

@app.route('/upload_video')
def upload_video_page():
    return render_template('upload_video.html')

@app.route('/use_webcam')
def use_webcam_page():
    return render_template('use_webcam.html')

@app.route('/livefeed')
def livefeed_page():
    return render_template('livefeed.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    filename = secure_filename(file.filename)
    
    temp_file_path = os.path.join('uploads', filename)
    file.save(temp_file_path)

    # Detect emotions
    emotions = detect_emotion_from_image(cv2.imread(temp_file_path))

    os.remove(temp_file_path)

    return jsonify(emotions)

@app.route('/upload_video', methods=['POST'])
def upload_video():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    filename = secure_filename(file.filename)

    temp_file_path = os.path.join('uploads', filename)
    file.save(temp_file_path)

    # Detect emotions from video
    frame_emotions = detect_emotion_from_video(temp_file_path)

    os.remove(temp_file_path)

    summary = {}
    for frame in frame_emotions:
        for emotion in frame:
            if emotion['emotion'] in summary:
                summary[emotion['emotion']] += 1
            else:
                summary[emotion['emotion']] = 1

    summary_output = [{'emotion': key, 'count': value} for key, value in summary.items()]

    return jsonify({
        'summary': summary_output,
        'details': frame_emotions
    })

@app.route('/detect_frame', methods=['POST'])
def detect_frame():
    data = request.get_json()
    img_data = data.get('image')

    img_data = img_data.split(',')[1]
    img_data = base64.b64decode(img_data)
    np_img = np.frombuffer(img_data, np.uint8)
    img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    emotions = detect_emotion_from_image(img)

    return jsonify(emotions)

if __name__ == '__main__':
    app.run(debug=True)
