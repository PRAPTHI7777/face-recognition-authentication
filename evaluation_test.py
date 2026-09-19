from app.evaluation import evaluate_thresholds, print_results


genuine_pairs = [
    ("data/test/prapthi1.jpg", "data/test/prapthi2.jpg"),
]

impostor_pairs = [
    ("data/test/prapthi1.jpg", "data/test/unknown.jpg"),
]

thresholds = [
    0.30,
    0.35,
    0.40,
    0.45,
    0.50,
    0.55,
    0.60,
    0.65,
    0.70,
]


results = evaluate_thresholds(
    genuine_pairs=genuine_pairs,
    impostor_pairs=impostor_pairs,
    thresholds=thresholds,
    ctx_id=-1,
)

print_results(results)
