"""FastAPI web app: prompt, reference audio, and scoring endpoints."""
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import FileResponse, Response

from proscor import audio, feedback, prompts, score as scorer
from proscor.tts import reference_bytes

STATIC_DIR = Path(__file__).resolve().parent / "static"

app = FastAPI(title="proscor-en")


@app.get("/api/prompt")
def api_prompt():
    p = prompts.get_prompt()
    return {"text": p["text"], "id": p["id"]}


@app.get("/api/reference")
def api_reference(text: str):
    wav_bytes = reference_bytes(text)
    return Response(content=wav_bytes, media_type="audio/wav")


@app.post("/api/score")
async def api_score(audio_file: UploadFile = File(..., alias="audio"), target_text: str = Form(...)):
    with tempfile.NamedTemporaryFile(suffix=".wav") as tmp:
        tmp.write(await audio_file.read())
        tmp.flush()
        samples, sr = audio.load_wav(tmp.name)

    report = scorer.score_audio(target_text, samples, sr=sr)
    report["feedback"] = feedback.format_report(report)
    return report


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")
