import os, re, shutil, subprocess, uuid
from pathlib import Path
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from openai import OpenAI

load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")
if not API_KEY:
    print("WARNING: OPENAI_API_KEY is not configured.")

client = OpenAI(api_key=API_KEY) if API_KEY else None
BASE = Path(__file__).resolve().parent
DATA = BASE / "data"
DATA.mkdir(exist_ok=True)
MAX_MB = int(os.getenv("MAX_UPLOAD_MB", "2048"))

app = FastAPI(title="RecapMM AI")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

jobs = {}

def sh(args):
    return subprocess.run(args, check=True, capture_output=True, text=True)

def srt_time(seconds):
    ms = int(max(0, seconds) * 1000)
    h, ms = divmod(ms, 3600000)
    m, ms = divmod(ms, 60000)
    s, ms = divmod(ms, 1000)
    return f"{h:02}:{m:02}:{s:02},{ms:03}"

def make_srt(text, duration, path):
    # Sentence-timed subtitles. This keeps the MVP simple and reliable.
    parts = [p.strip() for p in re.split(r"(?<=[.!?။！？])\s+|\n+", text) if p.strip()]
    if not parts:
        parts = [text.strip()]
    weights = [max(1, len(p)) for p in parts]
    total = sum(weights)
    t = 0.0
    rows = []
    for i, (p, w) in enumerate(zip(parts, weights), 1):
        d = duration * w / total
        rows.append(f"{i}\n{srt_time(t)} --> {srt_time(t+d)}\n{p}\n")
        t += d
    path.write_text("\n".join(rows), encoding="utf-8")

def media_duration(path):
    r = sh(["ffprobe","-v","error","-show_entries","format=duration","-of","default=nw=1:nk=1",str(path)])
    return float(r.stdout.strip())

def transcribe(audio):
    with audio.open("rb") as f:
        result = client.audio.transcriptions.create(
            model=os.getenv("OPENAI_TRANSCRIBE_MODEL","gpt-4o-transcribe"),
            file=f
        )
    return result.text

def recap(transcript):
    model = os.getenv("OPENAI_TEXT_MODEL","gpt-5.6-luna")
    prompt = f"""You are a Myanmar movie-recap narrator.
Create an original Burmese-language recap narration from the transcript below.
Do NOT translate line-by-line. Condense it into a coherent story.
Use natural spoken Burmese, easy to understand, with suspenseful but not exaggerated narration.
Do not invent major events that are absent from the transcript.
Avoid quoting dialogue verbatim.
Target about 450-700 Burmese words for a normal-length source segment.

TRANSCRIPT:
{transcript}
"""
    r = client.responses.create(model=model, input=prompt)
    return r.output_text.strip()

def tts(text, out):
    model = os.getenv("OPENAI_TTS_MODEL","gpt-4o-mini-tts")
    voice = os.getenv("OPENAI_TTS_VOICE","alloy")
    with out.open("wb") as f:
        speech = client.audio.speech.create(
            model=model,
            voice=voice,
            input=text,
            response_format="mp3"
        )
        speech.write_to_file(out)

def assemble(video, voice, srt, out):
    # Scale/position subtitle text and mix narration with original audio at low volume.
    vf = f"subtitles='{srt.as_posix().replace(chr(39), chr(92)+chr(39))}':force_style='FontName=Arial,FontSize=20,Outline=2,Shadow=1,Alignment=2,MarginV=40'"
    sh([
        "ffmpeg","-y",
        "-i",str(video),
        "-i",str(voice),
        "-filter_complex",
        "[0:a]volume=0.18[orig];[1:a]volume=1.0[voice];[orig][voice]amix=inputs=2:duration=shortest[a]",
        "-map","0:v:0","-map","[a]",
        "-vf",vf,
        "-c:v","libx264","-preset","veryfast","-crf","23",
        "-c:a","aac","-b:a","192k","-shortest",str(out)
    ])

@app.get("/api/health")
def health():
    return {"ok": True, "openai_configured": bool(client)}

@app.post("/api/upload")
async def upload(video: UploadFile = File(...)):
    if not video.filename:
        raise HTTPException(400, "No filename")
    ext = Path(video.filename).suffix.lower()
    if ext not in {".mp4",".mov",".mkv",".webm"}:
        raise HTTPException(400, "Use MP4, MOV, MKV or WEBM")
    job_id = uuid.uuid4().hex
    d = DATA/job_id
    d.mkdir()
    p = d/("input"+ext)
    size = 0
    with p.open("wb") as f:
        while chunk := await video.read(1024*1024):
            size += len(chunk)
            if size > MAX_MB*1024*1024:
                p.unlink(missing_ok=True)
                raise HTTPException(413, f"Maximum upload is {MAX_MB} MB")
            f.write(chunk)
    jobs[job_id] = {"status":"uploaded","progress":5,"message":"Uploaded"}
    return {"job_id":job_id}

@app.post("/api/generate/{job_id}")
def generate(job_id: str):
    if job_id not in jobs:
        raise HTTPException(404,"Job not found")
    if not client:
        raise HTTPException(500,"OPENAI_API_KEY is missing")
    d = DATA/job_id
    inputs = list(d.glob("input.*"))
    if not inputs:
        raise HTTPException(404,"Input not found")
    video = inputs[0]
    try:
        jobs[job_id] = {"status":"processing","progress":10,"message":"Extracting audio"}
        audio = d/"speech.wav"
        sh(["ffmpeg","-y","-i",str(video),"-vn","-ac","1","-ar","16000",str(audio)])

        jobs[job_id].update(progress=30,message="Transcribing")
        transcript = transcribe(audio)
        (d/"transcript.txt").write_text(transcript,encoding="utf-8")

        jobs[job_id].update(progress=50,message="Writing Myanmar recap")
        script = recap(transcript)
        (d/"recap.txt").write_text(script,encoding="utf-8")

        jobs[job_id].update(progress=70,message="Generating Myanmar voice")
        voice = d/"voice.mp3"
        tts(script, voice)

        jobs[job_id].update(progress=82,message="Creating subtitles")
        duration = media_duration(voice)
        srt = d/"subtitles.srt"
        make_srt(script,duration,srt)

        jobs[job_id].update(progress=90,message="Rendering final video")
        final = d/"recapmm-final.mp4"
        assemble(video,voice,srt,final)

        jobs[job_id].update(status="done",progress=100,message="Ready")
        return {"status":"done","download":f"/api/download/{job_id}"}
    except Exception as e:
        jobs[job_id] = {"status":"error","progress":0,"message":str(e)}
        raise HTTPException(500,str(e))

@app.get("/api/status/{job_id}")
def status(job_id: str):
    if job_id not in jobs:
        raise HTTPException(404,"Job not found")
    return jobs[job_id]

@app.get("/api/download/{job_id}")
def download(job_id: str):
    p = DATA/job_id/"recapmm-final.mp4"
    if not p.exists():
        raise HTTPException(404,"Final video is not ready")
    return FileResponse(p, media_type="video/mp4", filename="recapmm-final.mp4")
