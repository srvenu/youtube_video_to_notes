# Enhanced YouTube Video Summarizer with Notes

##Demo Video

You can watch the demo video of the application in action:

[![Watch the Demo Video](demo-thumbnail.jpg)](demo.gif)

**[Watch the Demo Video (MP4)](video.mp4)**

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

```
```bash
pip install -r requirements.txt
```
```bash
python -m spacy download en_core_web_sm
```

How to Run the Application

Clone this repository to your local machine:

```bash
git clone https://github.com/yourusername/enhanced-youtube-summarizer.git
cd enhanced-youtube-summarizer
```
Install the required packages (as mentioned above).
Run the Streamlit application:

```bash
streamlit run app.py
```
Open your web browser and go to the URL displayed in the terminal (usually http://localhost:8501).
Usage Instructions

Enter the YouTube video URL in the provided input field.
Click the "Summarize and Generate Notes" button.
View the fetched transcript, summary, and important topics.
Download the generated PDF for offline access.
