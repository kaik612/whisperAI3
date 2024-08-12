from flask import Flask, render_template, request, jsonify
import speech_recognition as sr
import whisper
import numpy as np
import torch

app = Flask(__name__)

# Load the Whisper model (consider loading this only once at app startup for efficiency)
model = whisper.load_model("small")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/transcribe', methods=['POST'])
def transcribe():
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400

    audio_file = request.files['audio']
    audio_data = audio_file.read()

    # Convert audio data to numpy array
    audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0

    # Transcribe audio using Whisper
    result = model.transcribe(audio_np, fp16=torch.cuda.is_available(), language="japanese")
    text = result['text'].strip()

    return jsonify({'transcription': text})

if __name__ == '__main__':
    app.run(debug=True)
