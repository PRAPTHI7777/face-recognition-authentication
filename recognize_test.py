from app.recognition import recognize

result = recognize(
    "data/test/unknown.jpg",
    threshold=0.5,
    ctx_id=-1
)

print(result)