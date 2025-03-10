# Image Comparison Analysis MVP

A Streamlit application for comparing images and detecting differences using AI-powered analysis.

## Features

- Upload and compare two images
- Visual highlighting of differences
- AI-powered detailed analysis of changes
- Configurable comparison parameters
- Responsive web interface

## Deployment Instructions

1. Fork this repository to your GitHub account
2. Go to [Streamlit Cloud](https://share.streamlit.io/)
3. Sign in with your GitHub account
4. Click "New app"
5. Select your forked repository
6. Set the main file path to `app.py`
7. Add your Mistral API key in the Streamlit Cloud secrets:
   ```toml
   MISTRAL_API_KEY = "your-api-key-here"
   ```
8. Click "Deploy!"

## Local Development

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Create a `.env` file with your Mistral API key:
   ```
   MISTRAL_API_KEY=your-api-key-here
   ```
4. Run the app:
   ```bash
   streamlit run app.py
   ```
