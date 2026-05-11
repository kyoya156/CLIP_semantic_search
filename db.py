import chromadb
from sentence_transformers import SentenceTransformer
from PIL import Image

# 1. Khởi tạo
model = SentenceTransformer('clip-ViT-B-32')
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_or_create_collection(name="image_search")

# 2. Thêm ảnh vào DB (Chỉ cần làm 1 lần)
image_path = "path/to/your/image.jpg"
image_features = model.encode(Image.open(image_path)).tolist()

collection.add(
    embeddings=[image_features],
    metadatas=[{"path": image_path}],
    ids=["img_001"]
)

# 3. Tìm kiếm bằng văn bản
query_text = "a dog running in the park"
query_embedding = model.encode(query_text).tolist()

results = collection.query(
    query_embeddings=[query_embedding],
    n_results=5
)
