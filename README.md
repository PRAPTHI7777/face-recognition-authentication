# Face Recognition Identification System

An image-based face recognition system that enrolls individuals and identifies new faces by comparing them against an enrolled database.

## Requirements Implemented

| Requirement | Implementation |
|---|---|
| Face Detection | InsightFace |
| Face Embeddings | 512-dimensional embeddings |
| Similarity Matching | Cosine similarity |
| Identity Matching | Highest similarity score |
| Unknown Rejection | Threshold-based (`0.50`) |
| Enrollment | Embedding stored as JSON |
| Evaluation | Genuine vs. impostor pairs |
| Failure Handling | No face, multiple faces, unknown identity |

## How It Works

<img width="1224" height="1285" alt="Face recognition workflow" src="https://github.com/user-attachments/assets/3c8b3140-388a-4d97-a856-f1095b52f790" />

### Enrollment

```text
Image → Face Detection → 512-D Embedding → Normalize → Store Name + Embedding
```

The system requires exactly one face in the enrollment image. The resulting embedding is stored in `data/enrolled/`.

### Recognition

```text
Image → Face Detection → 512-D Embedding
      → Compare with Enrolled Embeddings
      → Highest Similarity
      → Threshold 0.50
      → Matched Identity / Unknown
```

## Model Used

- **Model:** InsightFace `buffalo_l`
- **Embedding:** 512-dimensional
- **Runtime:** ONNX Runtime
- **Execution:** CPU
- **Libraries:** Python, InsightFace, OpenCV, NumPy

A pretrained model is used instead of training from scratch, allowing the project to focus on the complete identification and matching pipeline.

## Matching & Unknown Rejection

Face embeddings are compared using **cosine similarity**.

The configured threshold is:

```text
0.50
```

Decision rule:

```text
similarity >= 0.50  →  Matched
similarity <  0.50  →  Unknown
```

The threshold prevents low-similarity faces from being incorrectly assigned to an enrolled identity.

## Basic Evaluation

The evaluation uses:

- **Genuine pair:** `elon1.jpg` vs `elon2.jpg`
- **Impostor pair:** `elon1.jpg` vs `zuck.jpg`

A genuine pair contains images of the same person, while an impostor pair contains images of different people.

| Threshold | Genuine Accept | Impostor Accept | Accuracy |
|---:|---:|---:|---:|
| 0.30 | 100% | 0% | 100% |
| 0.35 | 100% | 0% | 100% |
| 0.40 | 100% | 0% | 100% |
| 0.45 | 100% | 0% | 100% |
| **0.50** | **100%** | **0%** | **100%** |
| 0.55 | 100% | 0% | 100% |
| 0.60 | 100% | 0% | 100% |
| 0.65 | 100% | 0% | 100% |
| 0.70 | 0% | 0% | 50% |

At the selected threshold of **0.50**, the genuine pair was accepted and the impostor pair was rejected.

At `0.70`, the genuine pair was rejected, demonstrating a false rejection caused by a threshold that was too high.

> **Evaluation note:** This is a small sanity-check evaluation containing one genuine pair and one impostor pair. The results should not be interpreted as real-world accuracy. A larger and more diverse dataset would be required for proper evaluation and threshold calibration.

## Failure Cases

The system handles the following cases:

- **No face detected** → returns `no_face`
- **Multiple faces detected** → recognition is rejected because exactly one face is required
- **Unknown person** → returns `Unknown` when similarity is below the threshold
- **Poor lighting, blur, extreme pose, or occlusion** → may reduce recognition reliability
- **Person not enrolled** → cannot be identified and may be returned as `Unknown`

## Example Results

### Known Person

```text
status: matched
identity: Elon
similarity: 1.0
```

### Unknown Person

```text
status: unknown
similarity: 0.21
```

### No Face

```text
status: no_face
message: No face detected in the image
```

## Project Structure

```text
face-recognition-system/
│
├── app/
│   ├── embedder.py       # Face detection and embeddings
│   ├── enrollment.py     # Enroll and store identities
│   ├── matcher.py        # Cosine similarity
│   ├── recognition.py    # Identify / reject faces
│   └── evaluation.py     # Threshold evaluation
│
├── data/
│   ├── enrolled/         # Stored face embeddings
│   └── test/             # Test images
│
├── enroll_test.py
├── recognize_test.py
├── evaluation_test.py
├── test_face.py
├── test_similarity.py
├── requirements.txt
└── README.md
```

## How to Run

### 1. Clone the repository

```powershell
git clone https://github.com/PRAPTHI7777/face-recognition-authentication.git
cd face-recognition-authentication
```

### 2. Create a virtual environment

```powershell
py -3.11 -m venv venv
.\venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```powershell
python -m pip install -r requirements.txt
```

The first run may download the InsightFace `buffalo_l` model.

### 4. Enroll a person

```powershell
python enroll_test.py
```

The generated embedding is stored in:

```text
data/enrolled/
```

### 5. Recognize a face

```powershell
python recognize_test.py
```

The system compares the input face against the enrolled database and returns either the matched identity or `Unknown`.

### 6. Run evaluation

```powershell
python evaluation_test.py
```

This evaluates the system across multiple similarity thresholds using genuine and impostor pairs.


## Possible Improvements

- Evaluate using a larger and more diverse dataset
- Calibrate the threshold using more genuine and impostor pairs
- Support multiple enrollment images per person
- Replace JSON storage with a database
- Add liveness detection
- Support multiple faces in one image
- Add a web/API interface
- Add FAR/FRR and additional evaluation metrics

## Limitations

This is an **image-based face identification prototype**, not a production biometric security system.

The evaluation dataset is small and the current implementation uses JSON-based storage. Production use would require larger-scale evaluation, threshold calibration, secure biometric storage, and additional anti-spoofing measures.
