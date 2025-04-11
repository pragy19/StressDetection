from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pandas as pd
import joblib
import xgboost as xgb
import imageio
import os
import logging
from scipy.signal import butter, filtfilt, find_peaks

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load model and scaler
model = None
scaler = None
try:
    model = xgb.XGBClassifier()
    model.load_model("last.json")
    scaler = joblib.load("last.pkl")
    logging.info("Model and scaler loaded successfully.")
except FileNotFoundError as e:
    logging.error(f"Error loading model or scaler: {e}")
except Exception as e:
    logging.error(f"Unexpected error loading model or scaler: {e}")

# Set video upload folder
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ------------------- Helper Functions ------------------- #

def bandpass_filter(signal, lowcut, highcut, fs, order=2):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, signal)

def extract_red_channel(video_path):
    try:
        reader = imageio.get_reader(video_path)
        red_values = []

        for frame in reader:
            if frame.ndim == 3:
                red_channel = frame[:, :, 0]
                avg_red = np.mean(red_channel)
                red_values.append(avg_red / 255.0)

        reader.close()
        return np.array(red_values)
    except Exception as e:
        logging.error(f"Error extracting red channel: {e}")
        return None

def calculate_hrv(peaks, fps):
    if len(peaks) < 2:
        return np.nan, np.nan, np.nan
    rr_intervals = np.diff(peaks) * (1000 / fps)
    mean_rr = np.mean(rr_intervals)
    sdrr = np.std(rr_intervals)
    rmssd = np.sqrt(np.mean(np.square(np.diff(rr_intervals))))
    return mean_rr, sdrr, rmssd

# ------------------- API Routes ------------------- #

@app.route('/')
def index():
    return "Stress Detection API is Running 🚀"

@app.route('/health')
def health():
    return jsonify({"status": "ok"}), 200

@app.route('/predict', methods=['POST'])
def predict_stress():
    if model is None or scaler is None:
        return jsonify({'error': 'Model or scaler not loaded'}), 500

    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']

    filename = os.path.join(UPLOAD_FOLDER, file.filename)
    try:
        file.save(filename)
        logging.info(f"Video saved to: {filename}")

        fps = 30.0  # Adjust if needed
        red_channel = extract_red_channel(filename)

        if red_channel is None or len(red_channel) < int(fps * 2):  # At least 2 seconds of data
            return jsonify({'error': 'Could not extract sufficient valid red channel data from the video'}), 400

        start_index = int(fps * 1)
        end_index = -int(fps * 1)
        red_channel_processed = red_channel[start_index:end_index]

        if len(red_channel_processed) < 10:
            return jsonify({'error': 'Insufficient data after processing'}), 400

        filtered_signal = bandpass_filter(red_channel_processed, 0.7, 2.5, fps)

        min_peak_distance = int(fps / 2.5)
        peak_height_threshold = np.mean(filtered_signal) * 0.8
        peaks, _ = find_peaks(filtered_signal, distance=min_peak_distance, height=peak_height_threshold)

        heart_rate = (len(peaks) / (len(filtered_signal) / fps)) * 60 if len(filtered_signal) > 0 else np.nan
        mean_rr, sdrr, rmssd = calculate_hrv(peaks, fps)

        input_data = [[mean_rr, sdrr, rmssd, heart_rate]]
        columns = ["MEAN_RR", "SDRR", "RMSSD", "HR"]
        input_df = pd.DataFrame(input_data, columns=columns)

        scaled_input = scaler.transform(input_df)

        prediction = model.predict(scaled_input)
        probabilities = model.predict_proba(scaled_input)[0]
        classes = ['no stress', 'interruption', 'time pressure']
        stress_class = classes[np.argmax(probabilities)]

        stress_percentage = (
            probabilities[classes.index('time pressure')] * 70 +
            probabilities[classes.index('interruption')] * 100
        )

        if stress_percentage < 30:
            stress_level = "Low Stress"
        elif stress_percentage < 60:
            stress_level = "Moderate Stress"
        elif stress_percentage < 85:
            stress_level = "High Stress"
        else:
            stress_level = "Critical Stress"

        return jsonify({
            "heart_rate": float(round(heart_rate, 2)) if not np.isnan(heart_rate) else None,
            "mean_rr": float(round(mean_rr, 2)) if not np.isnan(mean_rr) else None,
            "sdrr": float(round(sdrr, 2)) if not np.isnan(sdrr) else None,
            "rmssd": float(round(rmssd, 2)) if not np.isnan(rmssd) else None,
            "class": stress_class,
            "percentage": float(round(stress_percentage, 2)),
            "stress_level": stress_level
        })

    except Exception as e:
        logging.error(f"Error processing video: {e}")
        return jsonify({'error': str(e)}), 500
    finally:
        if os.path.exists(filename):
            os.remove(filename)
            logging.info(f"Deleted uploaded file: {filename}")

# ------------------- Run Server ------------------- #

import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Render provides PORT env var
    app.run(host='0.0.0.0', port=port, debug=True)

