from app.enrollment import enroll

result = enroll(
    "data/test/prapthi1.jpg",
    "Prapthi",
    ctx_id=-1
)

print(result)