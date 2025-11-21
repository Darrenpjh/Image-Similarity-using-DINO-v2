```markdown
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
│   └── collection/image_vectors/
│        ├── storage.sqlite
│        └── .lock
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

1. Install Python 3.8+  
2. Install dependencies:

```bash
pip install flask pillow qdrant-client torch transformers
```

---

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
```