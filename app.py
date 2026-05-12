"""
app.py — Semantic image search with CLIP + ChromaDB.

Run:
    streamlit run app.py
"""

import os
import io
import chromadb
import streamlit as st
from sentence_transformers import SentenceTransformer
from PIL import Image

# Config
CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "image_search"
MODEL_NAME = "clip-ViT-B-32"
TOP_K_MAX = 12


# Cached resources
@st.cache_resource
def load_model() -> SentenceTransformer:
    return SentenceTransformer(MODEL_NAME)


@st.cache_resource
def get_collection():
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    return client.get_collection(name=COLLECTION_NAME)


# Search helpers
def search_by_text(query: str, top_k: int, collection, model: SentenceTransformer):
    embedding = model.encode(query, convert_to_tensor=False).tolist()
    return collection.query(query_embeddings=[embedding], n_results=top_k)


def search_by_image(image: Image.Image, top_k: int, collection, model: SentenceTransformer):
    embedding = model.encode(image.convert("RGB"), convert_to_tensor=False).tolist()
    return collection.query(query_embeddings=[embedding], n_results=top_k)


def distance_to_score(distance: float) -> float:
    """Convert cosine distance (0–2) to a 0–100 similarity score."""
    return round((1 - distance / 2) * 100, 1)


def render_results(results: dict) -> None:
    ids = results["ids"][0]
    distances = results["distances"][0]
    metadatas = results["metadatas"][0]

    if not ids:
        st.warning("No results found.")
        return

    cols = st.columns(min(len(ids), 4))
    for i, (img_id, dist, meta) in enumerate(zip(ids, distances, metadatas)):
        col = cols[i % 4]
        path = meta.get("path", img_id)
        filename = meta.get("filename", os.path.basename(path))
        score = distance_to_score(dist)

        with col:
            try:
                st.image(Image.open(path), width='stretch')
            except Exception:
                st.error(f"Cannot load:\n{filename}")
            st.markdown(
                f"<div style='text-align:center'>"
                f"<b>{score}%</b><br>"
                f"<small style='color:gray'>{filename}</small>"
                f"</div>",
                unsafe_allow_html=True,
            )


# Page setup
st.set_page_config(
    page_title="Semantic Image Search",
    page_icon="",
    layout="wide",
)
st.title("Semantic Image Search")
st.caption("Powered by CLIP + ChromaDB")

# Load resources
model = load_model()

try:
    collection = get_collection()
    n_indexed = collection.count()
    st.sidebar.success(f" {n_indexed} image(s) indexed")
except Exception:
    st.error(
        "**ChromaDB collection not found.**\n\n"
        "Run `python db.py --images ./images` first to index your images."
    )
    st.stop()

# Sidebar controls
st.sidebar.header("Search settings")
top_k = st.sidebar.slider("Results to show", min_value=1, max_value=TOP_K_MAX, value=6)

# Search tabs
tab_text, tab_image = st.tabs(["Text search", "Image search"])
 
with tab_text:
    def _on_enter():
        st.session_state["do_search"] = True
 
    query = st.text_input(
        "Describe what you're looking for (in English):",
        placeholder="e.g. Sunny side up egg on toast",
        key="text_query",
        on_change=_on_enter,
    )
    clicked = st.button("Search", key="text_search")
 
    if (clicked or st.session_state.get("do_search")) and query.strip():
        st.session_state["do_search"] = False
        with st.spinner("Searching…"):
            results = search_by_text(query.strip(), top_k, collection, model)
        st.subheader(f"Top {top_k} results for: *{query}*")
        render_results(results)
 
with tab_image:
    uploaded = st.file_uploader(
        "Upload a query image:",
        type=["png", "jpg", "jpeg", "webp"],
    )
    if st.button("Search", key="image_search") and uploaded:
        query_image = Image.open(io.BytesIO(uploaded.read()))
        st.image(query_image, caption="Your query image", width=240)
        with st.spinner("Searching…"):
            results = search_by_image(query_image, top_k, collection, model)
        st.subheader(f"Top {top_k} visually similar images")
        render_results(results)