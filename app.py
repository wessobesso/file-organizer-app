from flask import Flask, request, jsonify
import os
import json
import base64
import mimetypes
import subprocess
from PyPDF2 import PdfReader
from PIL import Image
import pytesseract
import openai
import fitz  # PyMuPDF for PDF to text
import torch
import clip
import faiss
import tempfile
from werkzeug.utils import secure_filename
import trimesh
import docx
import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup
import shutil
import stat
import time
from pathlib import Path
import zipfile
from striprtf.striprtf import rtf_to_text
import numpy as np

# Setup
app = Flask(__name__)
openai.api_key = os.getenv("OPENAI_API_KEY")

# CLIP model for images
device = "cuda" if torch.cuda.is_available() else "cpu"
try:
    clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
    print("CLIP model loaded successfully")
except Exception as e:
    clip_model, clip_preprocess = None, None
    print(f"Warning: CLIP not loaded - image features disabled. Error: {str(e)}")

# FAISS vector DB
dimension = 512
index = faiss.IndexFlatL2(dimension)
file_index = []

UPLOAD_FOLDER = os.path.expanduser("~/Desktop/FileAIApp/uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
print(f"Upload folder set to: {UPLOAD_FOLDER}")

# File categories
FILE_CATEGORIES = {
    "documents": ["pdf", "docx", "txt", "rtf", "odt", "doc", "md"],
    "images": ["jpg", "jpeg", "png", "gif", "bmp", "tiff"],
    "videos": ["mp4", "mov", "avi", "mkv", "wmv"],
    "audio": ["mp3", "wav", "flac", "aac", "ogg"],
    "archives": ["zip", "rar", "7z", "tar", "gz"],
    "code": ["py", "js", "html", "css", "java", "cpp", "c", "h", "sh"],
    "data": ["csv", "json", "xml", "xlsx", "db", "sqlite"],
    "3d_models": ["stl", "obj", "fbx", "blend"],
    "ebooks": ["epub", "mobi"],
    "executables": ["exe", "dmg", "app", "msi", "deb"]
}

def get_file_category(extension):
    for category, exts in FILE_CATEGORIES.items():
        if extension.lower() in exts:
            return category
    return "other"

def extract_text_from_file(filepath):
    ext = os.path.splitext(filepath)[1].lower()
    try:
        if ext == ".pdf":
            doc = fitz.open(filepath)
            return "\n".join(page.get_text() for page in doc)
        elif ext in [".png", ".jpg", ".jpeg"]:
            if clip_model is None:
                return "(Image processing not available)"
            image = Image.open(filepath)
            return pytesseract.image_to_string(image)
        elif ext == ".stl":
            mesh = trimesh.load(filepath)
            return f"STL file with {len(mesh.vertices)} vertices and {len(mesh.faces)} faces."
        elif ext in [".txt", ".md"]:
            with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                return f.read()
        elif ext == ".docx":
            doc = docx.Document(filepath)
            return "\n".join([p.text for p in doc.paragraphs])
        elif ext == ".epub":
            book = epub.read_epub(filepath)
            text = ""
            for item in book.get_items():
                if item.get_type() == ebooklib.ITEM_DOCUMENT:
                    soup = BeautifulSoup(item.get_content(), 'html.parser')
                    text += soup.get_text()
            return text
        elif ext == ".rtf":
            with open(filepath, 'r') as f:
                return rtf_to_text(f.read())
        elif ext == ".odt":
            with zipfile.ZipFile(filepath) as z:
                with z.open('content.xml') as f:
                    return f.read().decode('utf-8')[:10000]
        elif ext == ".doc":
            with open(filepath, 'rb') as f:
                text = f.read().decode('utf-8', errors='ignore')
                return text[:10000]
        else:
            return "(Unsupported file type)"
    except Exception as e:
        return f"(Error extracting text: {e})"

def embed_text(text):
    response = openai.Embedding.create(
        input=text,
        model="text-embedding-ada-002"
    )
    return response["data"][0]["embedding"]

def embed_image(filepath):
    if clip_model is None:
        return np.zeros(dimension)
    image = Image.open(filepath).convert("RGB")
    image = clip_preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        return clip_model.encode_image(image).cpu().numpy()[0]

def add_file_to_index(path):
    ext = os.path.splitext(path)[1].lower()
    filename = os.path.basename(path)
    content = extract_text_from_file(path)

    if ext in [".jpg", ".jpeg", ".png"] and clip_model is not None:
        vector = embed_image(path)
    else:
        vector = embed_text(content)

    index.add(vector.reshape(1, -1))
    file_index.append({"path": path, "type": ext, "preview": content[:300]})

@app.route('/')
def home():
    return """
    <h1>File Organizer App</h1>
    <h2>How to use:</h2>
    
    <h3>1. Upload files:</h3>
    <code>curl -X POST -F "file=@yourfile.pdf" http://localhost:5000/upload</code>
    
    <h3>2. Search files:</h3>
    <code>curl -X POST -H "Content-Type: application/json" -d '{"query":"search term"}' http://localhost:5000/ask</code>
    
    <h3>3. List files:</h3>
    <a href="/files">/files</a> or <code>curl http://localhost:5000/files</code>
    """

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    try:
        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)
        add_file_to_index(filepath)
        
        return jsonify({
            "status": "success",
            "message": "File uploaded and indexed",
            "path": filepath,
            "type": os.path.splitext(filename)[1].lower()
        }), 200
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()
    if not data or 'query' not in data:
        return jsonify({"error": "No query provided"}), 400
    
    query = data['query']
    try:
        # Get embeddings for the query
        query_embedding = embed_text(query)
        
        # Search for similar files
        D, I = index.search(np.array([query_embedding]), 5)
        
        # Get top matches
        results = []
        for i in I[0]:
            if i < len(file_index):
                file_data = file_index[i]
                results.append({
                    "path": file_data["path"],
                    "preview": file_data["preview"],
                    "score": float(D[0][i])  # Convert numpy float to Python float
                })
        
        # Generate AI summary if there are results
        if results:
            summary = "\n\n".join(
                f"File: {os.path.basename(r['path'])}\nPreview: {r['preview']}"
                for r in results
            )
            
            ai_response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "Summarize these file results:"},
                    {"role": "user", "content": f"Query: {query}\nResults:\n{summary}"},
                ]
            )
            answer = ai_response["choices"][0]["message"]["content"]
        else:
            answer = "No matching files found"
        
        return jsonify({
            "query": query,
            "answer": answer,
            "results": results
        }), 200
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/files")
def list_files():
    return jsonify([{
        "path": f["path"],
        "type": f["type"],
        "preview": f["preview"][:100] + "..." if len(f["preview"]) > 100 else f["preview"]
    } for f in file_index])

if __name__ == "__main__":
    try:
        print("\n=== Starting File Indexing ===")
        doc_count = 0
        error_count = 0
        docs_path = os.path.expanduser("~/Documents")
        print(f"Indexing files in: {docs_path}")
        
        for root, dirs, files in os.walk(docs_path):
            for f in files:
                full_path = os.path.join(root, f)
                try:
                    add_file_to_index(full_path)
                    doc_count += 1
                    if doc_count % 100 == 0:
                        print(f"Indexed {doc_count} files...")
                except Exception as e:
                    error_count += 1
                    if error_count < 5:
                        print(f"! Skipped {f[:20]}...: {str(e)}")
        
        print(f"\nIndexing complete. Success: {doc_count}, Errors: {error_count}")
        print("Starting Flask server at http://localhost:5000")
        print("Press Ctrl+C to stop the server\n")
        
        app.run(port=5000, debug=True, use_reloader=False)
        
    except KeyboardInterrupt:
        print("\nServer stopped by user")
    except Exception as e:
        print(f"\n=== CRITICAL ERROR ===")
        print(f"Error: {str(e)}")
        print("="*30)
        raise
