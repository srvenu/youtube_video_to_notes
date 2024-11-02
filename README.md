# Enhanced YouTube Video Summarizer with Notes

This project provides a Streamlit application that allows users to summarize YouTube videos and extract important topics from their transcripts. It is designed to help users quickly understand the main points of video content and create notes for easier reference.

## Features

- **Video URL Input**: Enter any YouTube video URL to fetch its transcript.
- **Transcript Retrieval**: Automatically retrieves the transcript in English.
- **Summarization**: Uses advanced NLP techniques to summarize the transcript.
- **Topic Extraction**: Identifies important topics such as people, organizations, events, and products.
- **Detailed Topic Notes**: Generates notes for each identified topic.
- **PDF Generation**: Create and download a PDF containing the summarized notes and video thumbnail.

## Requirements

To run this application, you need to install the following Python packages:

```bash
streamlit==1.20.0
youtube-transcript-api==0.6.0
transformers==4.30.0
spacy==3.4.0
fpdf==1.7.2
requests==2.28.1
pytube==11.0.0
