import streamlit as st
from sentence_transformers import SentenceTransformer, util
from PIL import Image
import os
import torch

# 1. Tải mô hình CLIP
@st.cache_resource
def load_model():
    return SentenceTransformer('clip-ViT-B-32')

model = load_model()

# 2. Cấu hình giao diện
st.title("🔎 Semantic Image Search (CLIP)")
img_folder = 'images'  # Thư mục chứa ảnh của bạn
img_names = [f for f in os.listdir(img_folder) if f.endswith(('png', 'jpg', 'jpeg'))]

# 3. Mã hóa toàn bộ ảnh trong thư mục (Vectorization)
@st.cache_data
def encode_images(img_names):
    img_paths = [os.path.join(img_folder, name) for name in img_names]
    images = [Image.open(path) for path in img_paths]
    return model.encode(images, convert_to_tensor=True)

img_embeddings = encode_images(img_names)

# 4. Ô tìm kiếm
query = st.text_input("Nhập mô tả ảnh muốn tìm (bằng tiếng Anh):", "A photo of a cat")

if query:
    # Mã hóa câu truy vấn
    query_embedding = model.encode(query, convert_to_tensor=True)
    
    # Tính toán độ tương đồng (Cosine Similarity)
    hits = util.semantic_search(query_embedding, img_embeddings, top_k=3)[0]
    
    # Hiển thị kết quả
    st.subheader("Kết quả phù hợp nhất:")
    cols = st.columns(3)
    for i, hit in enumerate(hits):
        idx = hit['corpus_id']
        score = hit['score']
        image_path = os.path.join(img_folder, img_names[idx])
        
        with cols[i]:
            st.image(image_path, caption=f"Score: {score:.2f}")
            st.write(f"File: {img_names[idx]}")
