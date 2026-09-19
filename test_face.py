import cv2
from insightface.app import FaceAnalysis

print("Loading model...")

app = FaceAnalysis(name="buffalo_l")
app.prepare(ctx_id=0, det_size=(640, 640))

print("Model loaded!")

image = cv2.imread("test.jpg")

faces = app.get(image)

print("Faces detected:", len(faces))

for face in faces:
    print("Bounding box:", face.bbox)
    print("Embedding shape:", face.embedding.shape)
    print("First 5 embedding values:", face.embedding[:5])