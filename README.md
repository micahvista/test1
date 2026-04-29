Setup
Prerequisites
Python 3.8+

A modern web browser (Chrome, Firefox, Edge)

1. Backend Setup
Clone this repository and navigate to the project folder.

Install the required Python dependencies:

Bash
pip install fastapi uvicorn torch transformers librosa scipy pillow python-multipart
Note: The backend is configured to use the https://hf-mirror.com endpoint by default to speed up Hugging Face model downloads in restricted regions. The first run may take a few minutes to download the Wav2Vec 2.0 and ViT models.

Start the FastAPI server:

Bash
uvicorn app:app --host 127.0.0.1 --port 8000 --reload
2. Frontend Setup
Ensure the backend is running on 127.0.0.1:8000.

Open the sound.html file directly in your web browser.

Grant the browser permission to access your microphone and camera when prompted.
