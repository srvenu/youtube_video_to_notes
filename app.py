#This is important py file
import streamlit as st
import re
import os
import tempfile
from fpdf import FPDF
from youtube_transcript_api import YouTubeTranscriptApi, NoTranscriptFound
from transformers import pipeline
import spacy

try:
    nlp = spacy.load("en_core_web_sm")
except:
    import os
    os.system("python -m spacy download en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

# Initialize models
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
nlp = spacy.load("en_core_web_sm")

def extract_video_id(url):
    """Extract the video ID from a YouTube URL."""
    pattern = r"(?:https?:\/\/)?(?:www\.)?(?:youtube\.com\/(?:[^\/\n\s]+\/\S+\/|(?:v|e(?:mbed)?)\/|.*[?&]v=)|youtu\.be\/)([a-zA-Z0-9_-]{11})"
    match = re.search(pattern, url)
    return match.group(1) if match else None

def fetch_transcript(video_id):
    """Retrieve transcript for the specified video ID."""
    try:
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        return " ".join([entry['text'] for entry in transcript_list.find_transcript(['en-GB', 'en']).fetch()])
    except NoTranscriptFound:
        st.error("Transcript not found for this video.")
        return ""

def summarize_text(text, chunk_size=1000):
    """Generate a summarized version of the provided text."""
    chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
    summaries = [summarizer(chunk, max_length=200, min_length=50, do_sample=False)[0]['summary_text'] for chunk in chunks]
    return " ".join(summaries)

def extract_topics(summary):
    """Extract key topics from the summary using named entity recognition."""
    doc = nlp(summary)
    return list({ent.text for ent in doc.ents if ent.label_ in ["PERSON", "ORG", "GPE", "EVENT", "PRODUCT", "NORP"]})

def create_notes_with_images(topics, transcript, images_info):
    """Generate notes on extracted topics and include image references."""
    notes = {topic: " ".join([sent.text for sent in nlp(transcript).sents if topic in sent.text]) for topic in topics}
    for img_path, caption in images_info:
        notes[caption] = f"Image: {caption}"
    return notes

def save_notes_to_pdf(notes, video_url, images_info, output_path="topic_notes.pdf"):
    """Generate a PDF file from the provided notes and images."""
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, "YouTube Video:", ln=True)
    pdf.cell(200, 10, video_url, ln=True)

    # Add notes to PDF
    for topic, content in notes.items():
        pdf.cell(200, 10, txt=f"{topic}", ln=True)
        pdf.multi_cell(0, 10, txt=content)
        pdf.cell(200, 10, '', ln=True)

    # Add images to PDF
    for img_path, caption in images_info:
        pdf.cell(200, 10, txt=f"Caption: {caption}", ln=True)
        pdf.image(img_path, x=10, w=150)
        pdf.cell(200, 10, '', ln=True)

    pdf.output(output_path)

    # Clean up temporary image files
    for img_path, _ in images_info:
        os.remove(img_path)

    return output_path

# Streamlit App Layout
st.set_page_config(layout="wide")

# Title
st.title("Enhanced YouTube Video Summarizer with Notes")

# Input section for YouTube URL
youtube_url = st.text_input("Enter YouTube Video URL")

# Directory for temporary image storage
images_info = []
temp_img_dir = "temp_img"
os.makedirs(temp_img_dir, exist_ok=True)

# Demo video section (optional, local video example)
video_path = '/Users/venusraj/Documents/ytsum/video.mp4'
if os.path.exists(video_path):
    st.subheader("Demo Video Showcasing Project Functionality")
    st.video(video_path, format="video/mp4", start_time=0)

# If a YouTube URL is provided, process it
if youtube_url:
    video_id = extract_video_id(youtube_url)
    if video_id:
        video_url = f"https://www.youtube.com/watch?v={video_id}"

        # Display the video and allow image uploads
        col1, col2 = st.columns([1, 1])

        with col1:
            st.video(video_url)

        with col2:
            st.subheader("Upload Images and Enter Captions")
            uploaded_files = st.file_uploader("Upload images (optional)", type=["jpg", "png", "jpeg"], accept_multiple_files=True)
            if uploaded_files:
                for uploaded_file in uploaded_files:
                    with tempfile.NamedTemporaryFile(delete=False, dir=temp_img_dir, suffix=".jpg") as temp_file:
                        temp_file.write(uploaded_file.read())
                        image_path = os.path.join(temp_img_dir, uploaded_file.name)
                        os.rename(temp_file.name, image_path)
                        st.image(image_path, width=200)
                        caption = st.text_input(f"Enter caption for {uploaded_file.name}", key=uploaded_file.name)
                        images_info.append((image_path, caption))

        # Summarize and generate notes
        if st.button("Summarize and Generate Notes"):
            transcript = fetch_transcript(video_id)
            if transcript:
                st.text_area("Transcript", transcript, height=300)
                summary = summarize_text(transcript)
                st.subheader("Summary")
                st.write(summary)

                topics = extract_topics(summary)
                st.subheader("Important Topics")
                st.write(", ".join(topics))

                notes = create_notes_with_images(topics, transcript, images_info)
                pdf_file_path = save_notes_to_pdf(notes, video_url, images_info)

                with open(pdf_file_path, "rb") as pdf_file:
                    st.download_button(
                        label="Download Notes as PDF",
                        data=pdf_file,
                        file_name="topic_notes.pdf",
                        mime="application/pdf"
                    )
