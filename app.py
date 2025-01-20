from dotenv import load_dotenv
import csv
import os
from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import openai
import nltk
from nltk.corpus import cmudict
import string
from gtts import gTTS
import psycopg2


load_dotenv()

def get_db_connection():
    return psycopg2.connect(dsn=os.getenv("SUPABASE_DSN"))

# Load CMU Pronouncing Dictionary
nltk.download("cmudict")
cmu_dict = cmudict.dict()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

openai.api_key = os.getenv("OPENAI_API_KEY")

UPLOAD_FOLDER = 'generated_files'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

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

@app.post('/process-audio')
async def process_audio(file: UploadFile = File(...)):
    audio_path = os.path.join(UPLOAD_FOLDER, file.filename)
    with open(audio_path, "wb") as f:
        f.write(await file.read())

    raw_transcription = transcribe_audio(audio_path)
    corrected_transcription = enhance_text_with_openai(raw_transcription)
    enhanced_text = enhance_text_with_openai(corrected_transcription)

    comparison_data = compare_phonemes(raw_transcription, corrected_transcription)
    phoneme_csv_path = generate_csv(comparison_data, "phoneme_comparison.csv", ["Raw Word", "Raw Phonemes", "Corrected Phonemes", "Match"])

    enhanced_phoneme_data = [{"Enhanced Word": word, "Enhanced Phonemes": " ".join(extract_phonemes(word))} for word in enhanced_text.split()]
    enhanced_phoneme_csv_path = generate_csv(enhanced_phoneme_data, "enhanced_phonemes.csv", ["Enhanced Word", "Enhanced Phonemes"])

    return JSONResponse({
        "raw_transcription": raw_transcription,
        "corrected_transcription": corrected_transcription,
        "enhanced_text": enhanced_text,
        "phoneme_comparison_data": comparison_data,
        "enhanced_phoneme_data": enhanced_phoneme_data,
        "phoneme_comparison_csv": f"/download/phoneme_comparison.csv",
        "enhanced_phoneme_csv": f"/download/enhanced_phonemes.csv"
    })

@app.get('/download/{filename}')
async def download_file(filename: str):
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(filepath, filename=filename)

@app.get('/')
async def home():
    return {"message": "Welcome to the Speech Enhancer App API. Use /process-audio to process audio files. The backend is running perfectly."}

@app.post('/generate-audio')
async def generate_audio(request: Request):
    data = await request.json()
    text = data.get('text', '')

    if not text:
        raise HTTPException(status_code=400, detail="No text provided")

    try:
        tts = gTTS(text=text, lang='en')
        audio_filename = "enhanced_audio.mp3"
        audio_filepath = os.path.join(UPLOAD_FOLDER, audio_filename)
        tts.save(audio_filepath)
        return FileResponse(audio_filepath, filename=audio_filename)
    except Exception as e:
        print(f"Error during audio generation: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to generate audio: {str(e)}")

@app.get('/test-connection')
async def test_connection():
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT 1;")
        cur.close()
        conn.close()
        return {"status": "Connection successful!"}
    except Exception as e:
        print("Error connecting to the database:", str(e))
        raise HTTPException(status_code=500, detail=str(e))


