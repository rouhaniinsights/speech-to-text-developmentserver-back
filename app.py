from dotenv import load_dotenv
import csv
import os
from flask import Flask, request, jsonify, send_file, url_for
from flask_cors import CORS
import openai
import nltk
from nltk.corpus import cmudict
import string
from gtts import gTTS
# import psycopg2

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Set OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

# Directory to store generated CSV files
UPLOAD_FOLDER = 'generated_files'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load CMU Pronouncing Dictionary
nltk.download("cmudict")
cmu_dict = cmudict.dict()

# Database connection function
# def get_db_connection():
#     return psycopg2.connect(dsn=os.getenv("SUPABASE_DSN"))

# Utility functions
def clean_word(word):
    """Remove punctuation and convert to lowercase."""
    return word.translate(str.maketrans('', '', string.punctuation)).lower()

def extract_phonemes(word):
    """Extract phonemes for a single word."""
    cleaned_word = clean_word(word)
    return cmu_dict.get(cleaned_word, [["[Not found]"]])[0]

def compare_phonemes(raw_text, corrected_text):
    """Compare phonemes of raw and corrected text and generate comparison data."""
    raw_words = raw_text.split()
    corrected_words = corrected_text.split()
    comparison_data = []

    for raw_word, corrected_word in zip(raw_words, corrected_words):
        raw_phonemes = " ".join(extract_phonemes(raw_word))
        corrected_phonemes = " ".join(extract_phonemes(corrected_word))
        match = "Match" if raw_phonemes == corrected_phonemes else "Mismatch"
        comparison_data.append({
            "Raw Word": raw_word,
            "Raw Phonemes": raw_phonemes,
            "Corrected Phonemes": corrected_phonemes,
            "Match": match
        })
    return comparison_data

def generate_csv(data, filename, fieldnames):
    """Generate a CSV file."""
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    with open(filepath, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)
    return filepath

def transcribe_audio(audio_file_path):
    """Transcribe audio using OpenAI's Whisper API."""
    try:
        with open(audio_file_path, "rb") as audio:
            response = openai.Audio.transcribe("whisper-1", audio)
        return response["text"]
    except Exception as e:
        print(f"Error during transcription: {str(e)}")
        return f"Error in transcription: {str(e)}"

def enhance_text_with_openai(text):
    """Enhance text using OpenAI's GPT model."""
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": f"Please enhance the following text:\n{text}"}],
            max_tokens=200,
            temperature=0.7
        )
        return response['choices'][0]['message']['content'].strip()
    except Exception as e:
        print(f"Error during enhancement: {str(e)}")
        return f"Error in enhancement: {str(e)}"

# Flask routes
@app.route('/process-audio', methods=['POST'])
def process_audio():
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400

    audio_file = request.files['file']
    audio_path = os.path.join(UPLOAD_FOLDER, audio_file.filename)
    audio_file.save(audio_path)

    # Step 1: Transcribe the audio file
    raw_transcription = transcribe_audio(audio_path)

    # Step 2: Enhance the transcription
    corrected_transcription = enhance_text_with_openai(raw_transcription)
    enhanced_text = enhance_text_with_openai(corrected_transcription)

    # Step 3: Generate comparison data and CSV files
    comparison_data = compare_phonemes(raw_transcription, corrected_transcription)
    phoneme_csv_path = generate_csv(comparison_data, "phoneme_comparison.csv", ["Raw Word", "Raw Phonemes", "Corrected Phonemes", "Match"])

    enhanced_phoneme_data = [{
        "Enhanced Word": word,
        "Enhanced Phonemes": " ".join(extract_phonemes(word))
    } for word in enhanced_text.split()]
    enhanced_phoneme_csv_path = generate_csv(enhanced_phoneme_data, "enhanced_phonemes.csv", ["Enhanced Word", "Enhanced Phonemes"])

    return jsonify({
        "raw_transcription": raw_transcription,
        "corrected_transcription": corrected_transcription,
        "enhanced_text": enhanced_text,
        "phoneme_comparison_data": comparison_data,
        "enhanced_phoneme_data": enhanced_phoneme_data,
        "phoneme_comparison_csv": url_for('download_file', filename="phoneme_comparison.csv", _external=True),
        "enhanced_phoneme_csv": url_for('download_file', filename="enhanced_phonemes.csv", _external=True)
    })

@app.route('/download/<filename>', methods=['GET'])
def download_file(filename):
    """Endpoint to download generated CSV files."""
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    if not os.path.exists(filepath):
        return jsonify({"error": "File not found"}), 404
    return send_file(filepath, as_attachment=True)

@app.route('/', methods=['GET'])
def home():
    return "Welcome to the Speech Enhancer App API. Use /process-audio to process audio files. The backend is running perfectly."

@app.route('/generate-audio', methods=['POST'])
def generate_audio():
    """Endpoint to generate audio from enhanced text."""
    data = request.json
    text = data.get('text', '')
    if not text:
        return jsonify({"error": "No text provided"}), 400

    try:
        # Use gTTS to generate speech
        tts = gTTS(text=text, lang='en')
        audio_filename = "enhanced_audio.mp3"
        audio_filepath = os.path.join(UPLOAD_FOLDER, audio_filename)
        tts.save(audio_filepath)
        return send_file(audio_filepath, as_attachment=True, download_name=audio_filename)
    except Exception as e:
        print(f"Error during audio generation: {str(e)}")
        return jsonify({"error": f"Failed to generate audio: {str(e)}"}), 500

# @app.route('/test-connection', methods=['GET'])
# def test_connection():
#     try:
#         conn = get_db_connection()
#         cur = conn.cursor()
#         cur.execute("SELECT 1;")
#         cur.close()
#         conn.close()
#         return {"status": "Connection successful!"}, 200
#     except Exception as e:
#         print("Error connecting to the database:", str(e))
#         return {"error": str(e)}, 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
