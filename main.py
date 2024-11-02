import streamlit as st
import re
from youtube_transcript_api import YouTubeTranscriptApi, NoTranscriptFound
from transformers import pipeline
import spacy
from fpdf import FPDF
import requests
import os
from pytube import YouTube  # Make sure to import YouTube here

# Load models for summarization and NLP
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
nlp = spacy.load("en_core_web_sm")

def get_video_id(url):
    """Extract the video ID from the YouTube URL."""
    regex = r"(?:https?:\/\/)?(?:www\.)?(?:youtube\.com\/(?:[^\/\n\s]+\/\S+\/|(?:v|e(?:mbed)?)\/|.*[?&]v=)|youtu\.be\/)([a-zA-Z0-9_-]{11})"
    match = re.search(regex, url)
    return match.group(1) if match else None

# def get_video_title(video_id):
#     """Fetch the title of the YouTube video using the YouTube Data API."""
#     api_key = 'YOUR_API_KEY'  # Replace with your API key
#     url = f'https://www.googleapis.com/youtube/v3/videos?id={video_id}&key={api_key}&part=snippet'
#     try:
#         response = requests.get(url)
#         data = response.json()
#         if data['items']:
#             return data['items'][0]['snippet']['title']
#         else:
#             st.error("Could not fetch video title. Video may not exist.")
#             return "Video Title Unavailable"
#     except Exception as e:
#         st.error("Error fetching video title.")
#         return "Video Title Unavailable"

def get_transcript(video_id):
    """Fetch and return the transcript for a given video ID."""
    try:
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        transcript = transcript_list.find_transcript(['en-GB', 'en'])  # Attempt to find an English transcript
        return " ".join([entry['text'] for entry in transcript.fetch()])  # Fetch transcript
    except NoTranscriptFound:
        raise RuntimeError("No transcript found for this video in the specified languages.")
    except Exception as e:
        raise RuntimeError(f"Error fetching transcript: {e}")

def summarize_text(text):
    """Summarize the provided text."""
    max_chunk_size = 1000
    chunks = [text[i:i + max_chunk_size] for i in range(0, len(text), max_chunk_size)]
    
    summaries = []
    for chunk in chunks:
        summary = summarizer(chunk, max_length=200, min_length=50, do_sample=False)[0]['summary_text']
        summaries.append(summary)
    
    return " ".join(summaries)

def extract_important_topics(summary):
    """Extract important topics from the summary using NLP."""
    doc = nlp(summary)
    topics = {ent.text for ent in doc.ents if ent.label_ in ["PERSON", "ORG", "GPE", "EVENT", "PRODUCT", "NORP"]}
    return list(topics)

def generate_topic_notes(topics, full_text):
    """Generate information about extracted topics for user notes."""
    topic_notes = {}
    for topic in topics:
        context_sentences = [sent.text for sent in nlp(full_text).sents if topic in sent.text]
        topic_notes[topic] = " ".join(context_sentences)
    return topic_notes

def get_thumbnail_url(video_id):
    """Fetch the thumbnail URL for the YouTube video."""
    yt = YouTube(f"https://www.youtube.com/watch?v={video_id}")
    return yt.thumbnail_url

def create_pdf(topic_notes, video_thumbnail_url):
    """Create a PDF with the topic notes and video thumbnail."""
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    # Download the thumbnail image and save it locally
    response = requests.get(video_thumbnail_url)
    if response.status_code == 200:
        with open("thumbnail.jpg", "wb") as img_file:
            img_file.write(response.content)
        
        # Add the saved thumbnail to PDF
        pdf.image("thumbnail.jpg", x=10, w=100)  # Add thumbnail to PDF
    else:
        raise RuntimeError("Failed to download thumbnail image.")

    pdf.cell(200, 10, '', ln=True)  # Empty line after thumbnail

    for topic, notes in topic_notes.items():
        pdf.cell(200, 10, txt=f"{topic}", ln=True, align='L')
        pdf.multi_cell(0, 10, txt=notes)
        pdf.cell(200, 10, '', ln=True)  # Empty line between topics

    pdf_file_path = "topic_notes.pdf"
    pdf.output(pdf_file_path)

    # Clean up the temporary image file
    os.remove("thumbnail.jpg")

    return pdf_file_path

# Streamlit app configuration
st.title("Enhanced YouTube Video Summarizer with Notes")
youtube_url = st.text_input("Enter YouTube Video URL")

if st.button("Summarize and Generate Notes"):
    video_id = get_video_id(youtube_url)
    if video_id:
        try:
            video_thumbnail_url = get_thumbnail_url(video_id)
            # video_title = get_video_title(video_id)  # Fetch video title

            # Show thumbnail and title at the start
            st.image(video_thumbnail_url)  # Show title as caption

            # Fetch transcript
            transcript = get_transcript(video_id)
            st.text_area("Transcript", transcript, height=300)

            # Summarize transcript
            summary = summarize_text(transcript)
            st.subheader("Summary")
            st.write(summary)

            # Extract important topics
            important_topics = extract_important_topics(summary)
            st.subheader("Important Topics")
            st.write(", ".join(important_topics))

            if important_topics:
                # Generate topic notes
                topic_notes = generate_topic_notes(important_topics, transcript)

                # Create PDF
                pdf_file_path = create_pdf(topic_notes, video_thumbnail_url)

                # PDF download
                with open(pdf_file_path, "rb") as pdf_file:
                    st.download_button(
                        label="Download Notes as PDF",
                        data=pdf_file,
                        file_name="topic_notes.pdf",
                        mime="application/pdf"
                    )
        except Exception as e:
            st.error(f"Error: {e}")
    else:
        st.warning("Invalid YouTube URL. Please check the link.")
