import streamlit as st
import re
from youtube_transcript_api import YouTubeTranscriptApi
from transformers import pipeline
import spacy

# Load models for summarization and NLP
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
nlp = spacy.load("en_core_web_sm")

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

def extract_important_topics(summary):
    """Extract important topics from the summary using NLP."""
    doc = nlp(summary)
    topics = set()
    for ent in doc.ents:
        if ent.label_ in ["PERSON", "ORG", "GPE", "EVENT", "PRODUCT", "NORP"]:
            topics.add(ent.text)
    return list(topics)

def generate_topic_notes(topics, full_text):
    """Generate information about extracted topics for user notes."""
    topic_notes = {}
    for topic in topics:
        context_sentences = [sent.text for sent in nlp(full_text).sents if topic in sent.text]
        topic_notes[topic] = " ".join(context_sentences)
    return topic_notes

# Streamlit app configuration
st.title("Enhanced YouTube Video Summarizer with Notes")
youtube_url = st.text_input("Enter YouTube Video URL")

if st.button("Summarize and Generate Notes"):
    video_id = get_video_id(youtube_url)
    if video_id:
        try:
            transcript = get_transcript(video_id)
            st.text_area("Transcript", transcript, height=300)

            summary = summarize_text(transcript)
            st.subheader("Summary")
            st.write(summary)

            important_topics = extract_important_topics(summary)
            st.subheader("Important Topics")
            st.write(", ".join(important_topics))

            if important_topics:
                st.subheader("Readable Notes")
                topic_notes = generate_topic_notes(important_topics, transcript)
                for topic, notes in topic_notes.items():
                    st.markdown(f"### {topic}")
                    st.write(notes)
        except Exception as e:
            st.error(f"Error: {e}")
    else:
        st.warning("Invalid YouTube URL. Please check the link.")
