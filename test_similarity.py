import cv2
from insightface.app import FaceAnalysis
from app.matcher import cosine_similarity


app = FaceAnalysis(name="buffalo_l")
app.prepare(ctx_id=0, det_size=(640, 640))

image = cv2.imread("test.jpg")
faces = app.get(image)

embedding = faces[0].embedding

similarity = cosine_similarity(embedding, embedding)

print("Similarity of face with itself:", similarity)