
# Similarity Image Search App

A professional, engineering-friendly dashboard for searching image similarity with deep learning (DINOv2) and vector database (Qdrant), featuring an interactive web app and batch download tools.

---

## Project Structure
```
Similarity/
├── dinov2-base/           # DINOv2 model files/config
│   ├── config.json
│   ├── preprocessor_config.json
│   ├── pytorch_model.bin
│   └── README.md
├── images/                # Your image data
├── qdrant_storage/        # Local Qdrant vector DB
│   ├── collection/image_vectors/
│   │   ├── storage.sqlite
│   │   └── .lock
│   └── meta.json
├── static/
│   ├── script.js          # JS (image selection logic)
│   └── style.css          # CSS (dashboard/gallery)
├── templates/
│   └── index.html         # Main web page (Flask)
├── app.py                 # Main Flask web server
├── index_images.py        # Script for batch indexing
├── search_images.py       # CLI search tool
├── utils.py               # Shared embedding/model functions
├── embeddings.npy         # (optional) Saved vectors
├── img_paths.txt          # (optional) Paths reference
└── readme.md              # Project docs (this file)
```
---

## Features

- DINOv2 deep learning embeddings for images  
- Qdrant vector database for rapid similarity search  
- Responsive Flask/Bootstrap web UI with an engineering look  
- Upload or select images for querying  
- Large, clear gallery card results  
- Drag-to-select and click-to-select for batch actions  
- Multi-download as ZIP – select images with click/drag, get them in one zip file  

---

## Installation

1. Install Python 3.8 or higher (if not already installed).

2. Create and activate a virtual environment (recommended):

On Windows (PowerShell):

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```
On Windows (Command Prompt):

```cmd
python -m venv venv
venv\Scripts\activate.bat
```
On Linux/macOS:

```bash
python3 -m venv venv
source venv/bin/activate
```
3. Install required Python packages inside the virtual environment:

```bash
pip install --upgrade pip
pip install flask pillow qdrant-client torch transformers tqdm
```
4. Download DINOv2 model files from DINOv2 base model on Hugging Face and place them in the dinov2-base/ folder. (https://huggingface.co/facebook/dinov2-base/tree/main)

Notes:
Using a virtual environment keeps your project dependencies isolated from system-wide Python packages.

To deactivate the virtual environment, simply run:

```bash
deactivate
```

## Getting Started

### 1. Prepare Data  
Place images (`.jpg`, `.jpeg`, `.png`, `.tif`) inside the `images/` folder.

### 2. Index Images  
Embed images and create the vector database by running:

```bash
python index_images.py
```

### 3. Launch the Web App  
Start the Flask dashboard:

```bash
python app.py
```

Open your browser and visit: [http://localhost:8501](http://localhost:8501)

### 4. Usage  
- Select a query image (existing or upload)  
- Set the number of search results (`top_k`)  
- Select similar images with click or drag  
- Download batch results as a ZIP file  

---

## Customization

- Modify `static/style.css` and `templates/index.html` for styling and UI changes  
- Adjust `static/script.js` for selection features customization  
- Optional: add CSV export, metadata, user logins, or analytics  

---

## Troubleshooting

- Only run one Qdrant-based script (web app, `index_images.py`, etc.) at a time unless using Qdrant Server  
- For drag-to-select, use a desktop with a mouse  
- Missing images? Double-check filenames and formats inside the `images/` folder  

---

## Credits

- **DINOv2** – Embedding model  
- **Qdrant** – Vector search database  
- **Flask / Bootstrap** – Web dashboard framework  

---

## Contributing

By Darren 

