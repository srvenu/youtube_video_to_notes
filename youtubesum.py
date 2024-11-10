#This is not final py file
import streamlit as st
import re
from youtube_transcript_api import YouTubeTranscriptApi
from transformers import pipeline

# Load the model for summarization
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

def get_video_id(url):
    """Extract the video ID from the YouTube URL."""
    regex = r"(?:https?:\/\/)?(?:www\.)?(?:youtube\.com\/(?:[^\/\n\s]+\/\S+\/|(?:v|e(?:mbed)?)\/|.*[?&]v=)|youtu\.be\/)([a-zA-Z0-9_-]{11})"
    match = re.search(regex, url)
    if match:
        return match.group(1)
    else:
        return None

def get_transcript(video_id):
    """Fetch and return the transcript for a given video ID."""
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        return " ".join([entry['text'] for entry in transcript])
    except Exception as e:
        raise RuntimeError(f"Error fetching transcript: {e}")

def summarize_text(text):
    """Summarize the provided text."""
    max_chunk_size = 1000  # Adjust as needed
    chunks = [text[i:i + max_chunk_size] for i in range(0, len(text), max_chunk_size)]
    
    # Generate summary for each chunk
    summaries = []
    for chunk in chunks:
        summary = summarizer(chunk, max_length=200, min_length=50, do_sample=False)[0]['summary_text']
        summaries.append(summary)
    
    # Combine the summaries
    return " ".join(summaries)

# Streamlit app configuration
st.title("Local YouTube Video Summarizer")
youtube_url = st.text_input("Enter YouTube Video URL")

if st.button("Summarize"):
    video_id = get_video_id(youtube_url)
    if video_id:
        try:
            transcript = get_transcript(video_id)
            st.text_area("Transcript", transcript, height=300)

            summary = summarize_text(transcript)
            st.subheader("Summary")
            st.write(summary)
        except Exception as e:
            st.error(f"Error: {e}")
    else:
        st.warning("Invalid YouTube URL. Please check the link.")
