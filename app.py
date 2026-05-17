#v0.2.0
import asyncio
import json
import re
import sys
import threading
import unicodedata
from pathlib import Path
from typing import AsyncGenerator, List, Optional

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse, FileResponse
from fastapi.templating import Jinja2Templates
from fpdf import FPDF
from pydantic import BaseModel, Field
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import NoTranscriptFound, TranscriptsDisabled
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ---------------------------------------------------------------------------
BASE = Path(__file__).parent
TEMP_DIR = BASE / "temp_img"
TEMP_DIR.mkdir(exist_ok=True)

MODELS = ["llama3.1:8b", "qwen2.5:7b", "mistral:7b-instruct", "llama3.2:3b"]

app = FastAPI(title="YT Notes")
templates = Jinja2Templates(directory=str(BASE / "templates"))


# ---------------------------------------------------------------------------
# Pydantic output schema
# ---------------------------------------------------------------------------

class TopicNote(BaseModel):
    topic: str = Field(description="The topic name or key concept covered in the video")
    summary: str = Field(description="2-3 sentence summary of how this topic is discussed in the video")
    key_points: List[str] = Field(description="3-5 specific key points, insights, or facts about this topic")
    quotes: List[str] = Field(description="1-3 notable quotes or close paraphrases from the video on this topic")


class VideoNotes(BaseModel):
    title_guess: str = Field(description="Inferred title or subject of the video based on its content")
    overall_summary: str = Field(description="Comprehensive 4-6 sentence summary of the entire video")
    key_takeaways: List[str] = Field(description="5-7 most important takeaways the viewer should remember")
    topics: List[TopicNote] = Field(description="Detailed notes organised by topic — identify 3-6 distinct main topics")
    conclusion: str = Field(description="The video's concluding message, recommendation, or call to action")


# ---------------------------------------------------------------------------
# Pipeline helpers (run in background thread)
# ---------------------------------------------------------------------------

def extract_video_id(url: str) -> Optional[str]:
    pattern = (
        r"(?:https?:\/\/)?(?:www\.)?"
        r"(?:youtube\.com\/(?:[^\/\n\s]+\/\S+\/|(?:v|e(?:mbed)?)\/|.*[?&]v=)"
        r"|youtu\.be\/)([a-zA-Z0-9_-]{11})"
    )
    m = re.search(pattern, url)
    return m.group(1) if m else None


def fetch_transcript(video_id: str) -> str:
    api = YouTubeTranscriptApi()
    lst = api.list(video_id)
    t = lst.find_transcript(["en-GB", "en"])
    return " ".join(snippet.text for snippet in t.fetch())


def _safe(text: str) -> str:
    return unicodedata.normalize("NFKD", str(text)).encode("latin-1", "replace").decode("latin-1")


def build_pdf(notes: VideoNotes, video_url: str, images_info: List[tuple]) -> str:
    out = str(TEMP_DIR / "structured_notes.pdf")
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    pdf.set_font("Arial", "B", 17)
    pdf.multi_cell(0, 10, _safe(notes.title_guess), align="C")
    pdf.ln(2)
    pdf.set_font("Arial", size=9)
    pdf.set_text_color(110, 110, 110)
    pdf.cell(0, 7, _safe(video_url), ln=True)
    pdf.set_text_color(0, 0, 0)
    pdf.ln(4)

    def section(title: str):
        pdf.set_font("Arial", "B", 13)
        pdf.cell(0, 9, title, ln=True)

    section("Overview")
    pdf.set_font("Arial", size=11)
    pdf.multi_cell(0, 7, _safe(notes.overall_summary))
    pdf.ln(5)

    section("Key Takeaways")
    pdf.set_font("Arial", size=11)
    for i, t in enumerate(notes.key_takeaways, 1):
        pdf.multi_cell(0, 7, _safe(f"  {i}. {t}"))
    pdf.ln(5)

    section("Topic Notes")
    pdf.ln(2)
    matched = {tn.topic.lower() for tn in notes.topics}
    for tn in notes.topics:
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 9, _safe(f"  {tn.topic}"), ln=True)
        pdf.set_font("Arial", size=11)
        pdf.multi_cell(0, 7, _safe(tn.summary))
        pdf.set_font("Arial", "BI", 11)
        pdf.cell(0, 8, "  Key Points:", ln=True)
        pdf.set_font("Arial", size=11)
        for p in tn.key_points:
            pdf.multi_cell(0, 7, _safe(f"    • {p}"))
        if tn.quotes:
            pdf.set_font("Arial", "BI", 11)
            pdf.cell(0, 8, "  Notable Quotes:", ln=True)
            pdf.set_font("Arial", "I", 10)
            pdf.set_text_color(70, 70, 70)
            for q in tn.quotes:
                pdf.multi_cell(0, 7, _safe(f'    "{q}"'))
            pdf.set_text_color(0, 0, 0)
        for img_path, caption in images_info:
            if caption.lower() == tn.topic.lower():
                pdf.image(img_path, x=15, w=140)
        pdf.ln(5)

    section("Conclusion")
    pdf.set_font("Arial", size=11)
    pdf.multi_cell(0, 7, _safe(notes.conclusion))
    pdf.ln(5)

    for img_path, caption in images_info:
        if caption.lower() not in matched:
            pdf.set_font("Arial", "B", 11)
            pdf.cell(0, 9, _safe(f"Image: {caption}"), ln=True)
            pdf.image(img_path, x=15, w=140)
            pdf.ln(4)

    pdf.output(out)
    return out


# ---------------------------------------------------------------------------
# Background pipeline with SSE event queue
# ---------------------------------------------------------------------------

def run_pipeline(
    url: str,
    model: str,
    chunk_size: int,
    images_data: list,
    queue: asyncio.Queue,
    loop: asyncio.AbstractEventLoop,
):
    def emit(**kwargs):
        asyncio.run_coroutine_threadsafe(queue.put(json.dumps(kwargs)), loop)

    def done():
        asyncio.run_coroutine_threadsafe(queue.put(None), loop)

    try:
        video_id = extract_video_id(url)
        if not video_id:
            emit(type="error", message="Invalid YouTube URL — could not extract video ID.")
            return done()

        # ── 1. Transcript ────────────────────────────────────────────────────
        emit(type="progress", step="transcript", message="Fetching transcript…")
        try:
            transcript = fetch_transcript(video_id)
        except (NoTranscriptFound, TranscriptsDisabled):
            emit(type="error", message="No English transcript available for this video.")
            return done()
        except Exception as e:
            emit(type="error", message=f"Transcript error: {e}")
            return done()

        emit(type="transcript_preview", text=transcript[:600] + "…" if len(transcript) > 600 else transcript)

        # ── 2. Map-reduce summarisation ──────────────────────────────────────
        llm = ChatOllama(model=model, temperature=0)
        splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=200)
        chunks = splitter.split_text(transcript)

        map_prompt = ChatPromptTemplate.from_template(
            "Summarise the key points from this transcript section in 4-6 clear sentences:\n\n{text}"
        )
        reduce_prompt = ChatPromptTemplate.from_template(
            "You have partial summaries of a YouTube video. "
            "Combine them into one coherent, comprehensive summary (8-12 sentences):\n\n{text}\n\nFinal summary:"
        )
        map_chain = map_prompt | llm | StrOutputParser()
        reduce_chain = reduce_prompt | llm | StrOutputParser()

        partials: List[str] = []
        for i, chunk in enumerate(chunks):
            emit(type="progress", step="summarize", message=f"Summarising chunk {i+1} of {len(chunks)}…",
                 current=i + 1, total=len(chunks))
            partials.append(map_chain.invoke({"text": chunk}))

        emit(type="progress", step="summarize", message="Combining summaries…", current=len(chunks), total=len(chunks))
        summary = reduce_chain.invoke({"text": "\n\n---\n\n".join(partials)})

        # ── 3. Structured notes ──────────────────────────────────────────────
        emit(type="progress", step="notes", message="Extracting structured notes with Pydantic schema…")
        structured_llm = llm.with_structured_output(VideoNotes)
        notes_prompt = ChatPromptTemplate.from_template(
            "You are an expert note-taker and educator.\n"
            "Based on the video summary and transcript excerpt below, produce thorough structured notes.\n\n"
            "VIDEO SUMMARY:\n{summary}\n\n"
            "TRANSCRIPT EXCERPT (first 3000 chars):\n{excerpt}\n\n"
            "Return well-organised notes so someone can fully understand the video without watching it."
        )
        notes: VideoNotes = (notes_prompt | structured_llm).invoke(
            {"summary": summary, "excerpt": transcript[:3000]}
        )

        # ── 4. PDF ───────────────────────────────────────────────────────────
        emit(type="progress", step="pdf", message="Building PDF…")
        images_info = []
        for i, (content, caption) in enumerate(images_data):
            p = str(TEMP_DIR / f"upload_{i}.jpg")
            with open(p, "wb") as f:
                f.write(content)
            images_info.append((p, caption))

        build_pdf(notes, f"https://www.youtube.com/watch?v={video_id}", images_info)
        emit(type="result", notes=notes.model_dump())

    except Exception as e:
        emit(type="error", message=str(e))
    finally:
        done()


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(request=request, name="index.html", context={"models": MODELS})


@app.post("/generate")
async def generate(request: Request):
    form = await request.form()
    url = str(form.get("url", ""))
    model = str(form.get("model", "llama3.1:8b"))
    chunk_size = int(form.get("chunk_size", 3000))

    images_data = []
    for img_file, caption in zip(form.getlist("images"), form.getlist("captions")):
        if hasattr(img_file, "read"):
            images_data.append((await img_file.read(), str(caption)))

    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue()

    threading.Thread(
        target=run_pipeline,
        args=(url, model, chunk_size, images_data, queue, loop),
        daemon=True,
    ).start()

    async def event_stream() -> AsyncGenerator[str, None]:
        while True:
            msg = await queue.get()
            if msg is None:
                break
            yield f"data: {msg}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"X-Accel-Buffering": "no", "Cache-Control": "no-cache", "Connection": "keep-alive"},
    )


@app.get("/download")
async def download():
    p = TEMP_DIR / "structured_notes.pdf"
    if not p.exists():
        return {"error": "PDF not ready"}
    return FileResponse(str(p), media_type="application/pdf", filename="structured_notes.pdf")


if __name__ == "__main__":
    import uvicorn
    # Windows requires ProactorEventLoop for asyncio subprocess support
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
