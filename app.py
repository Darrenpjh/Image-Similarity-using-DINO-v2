import os
import io
from PIL import Image
from flask import Flask, render_template, request, send_from_directory, send_file, url_for
from qdrant_client import QdrantClient
from utils import load_dinov2, compute_embedding
import zipfile

IMAGES_FOLDER = "./images"
QDRANT_PATH = "./qdrant_storage"
COLLECTION_NAME = "image_vectors"
TOP_K = 5

app = Flask(__name__, static_folder="static")
processor, model, device = load_dinov2()
qdrant = QdrantClient(path=QDRANT_PATH)

def qdrant_search(query_vector, limit=5):
    hits = qdrant.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_vector,
        limit=limit
    )
    return hits

def get_image_files():
    return [f for f in os.listdir(IMAGES_FOLDER) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.tif'))]

@app.route("/", methods=["GET", "POST"])
def index():
    img_files = get_image_files()
    top_k = TOP_K
    results = []
    query_img_url = None
    msg = None

    if request.method == "POST":
        top_k = int(request.form.get("top_k", TOP_K))
        img_query = None
        if 'db_image' in request.form and request.form['db_image']:
            fname = request.form['db_image']
            query_path = os.path.join(IMAGES_FOLDER, fname)
            img_query = query_path   # pass as path (string)
            query_img_url = url_for('serve_img', filename=fname)
        elif 'query_img' in request.files and request.files['query_img']:
            file = request.files['query_img']
            image_bytes = file.read()
            pil_img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
            img_query = pil_img     # pass as PIL.Image
            temp_img_name = "static_query_temp.png"
            pil_img.save(os.path.join(IMAGES_FOLDER, temp_img_name))
            query_img_url = url_for('serve_img', filename=temp_img_name)
        else:
            msg = "Please select or upload a query image."
            return render_template("index.html", img_files=img_files, results=results, top_k=top_k, query_img_url=None, msg=msg)
        vec_query = compute_embedding(img_query, processor, model, device)
        hits = qdrant_search(vec_query, limit=top_k)
        results = []
        for hit in hits:
            fname = hit.payload.get("filename")
            sim_score = hit.score
            if fname and os.path.exists(os.path.join(IMAGES_FOLDER, fname)):
                img_url = url_for('serve_img', filename=fname)
                results.append({"img_url": img_url, "fname": fname, "score": sim_score})

    return render_template("index.html", img_files=img_files, results=results, top_k=top_k, query_img_url=query_img_url, msg=msg)

@app.route('/img/<path:filename>')
def serve_img(filename):
    return send_from_directory(IMAGES_FOLDER, filename)

@app.route("/download_selected", methods=["POST"])
def download_selected():
    fnames = request.form.getlist("selected_images")
    if not fnames:
        return "No images selected.", 400
    memory_file = io.BytesIO()
    with zipfile.ZipFile(memory_file, 'w') as zf:
        for fname in fnames:
            img_path = os.path.join(IMAGES_FOLDER, fname)
            if os.path.exists(img_path):
                zf.write(img_path, fname)
    memory_file.seek(0)
    return send_file(
        memory_file,
        mimetype='application/zip',
        as_attachment=True,
        download_name='selected_images.zip'
    )

import atexit
@atexit.register
def cleanup():
    temp_img_path = os.path.join(IMAGES_FOLDER, "static_query_temp.png")
    try:
        os.remove(temp_img_path)
    except Exception:
        pass

if __name__ == "__main__":
    app.run(debug=False, port=8505, threaded=False)
