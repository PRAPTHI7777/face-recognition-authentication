# Face Recognition Identification System

## Overview

This project is a simple face recognition identification system built for an
AI/ML internship assignment. It enrolls known individuals, extracts face
embeddings from images, and identifies new faces by comparing them with the
enrolled embedding database.

The implementation is intentionally small and explainable. It is not intended
to claim 100% accuracy in real-world conditions.

## How It Works

1. **Face detection** - InsightFace detects faces in an input image.
2. **Face embedding extraction** - The detected face is converted into a
   numerical feature vector.
3. **Normalization** - The embedding is L2-normalized before comparison.
4. **Cosine similarity matching** - The normalized input embedding is compared
   with every enrolled embedding.
5. **Unknown-face rejection** - The highest similarity is compared with a
   configurable threshold. If it is below the threshold, the result is
   returned as `Unknown`.

Enrollment and recognition currently require exactly one face in the image.

## Model

- **Model:** InsightFace `buffalo_l`
- **Embedding size:** 512 dimensions
- **Runtime:** ONNX Runtime
- **Image processing:** OpenCV
- **Numerical processing:** NumPy

## Enrollment Workflow

Enrollment accepts an image and a person name:

1. Load the image.
2. Detect faces.
3. Require exactly one face.
4. Extract and normalize the 512-dimensional embedding.
5. Save the name and embedding as a JSON file under `data/enrolled/`.

Example:

```python
from app.enrollment import enroll

enroll(
    "data/test/prapthi1.jpg",
    "Prapthi",
    ctx_id=-1,  # CPU; use 0 for GPU when configured
)
```

An existing enrollment is not overwritten unless `overwrite=True` is passed.

## Recognition Workflow

Recognition accepts an image path and a threshold:

1. Detect faces and require exactly one face.
2. Extract and normalize the input embedding.
3. Load all JSON enrollments from `data/enrolled/`.
4. Calculate cosine similarity against each enrolled embedding.
5. Select the highest similarity.
6. Return the matching identity when the threshold is met; otherwise return
   `Unknown`.

Example:

```python
from app.recognition import recognize

result = recognize(
    "data/test/prapthi2.jpg",
    threshold=0.50,
    ctx_id=-1,  # CPU; use 0 for GPU when configured
)

print(result)
```

## Matching Threshold

The current matching threshold is **0.50**:

```python
result = recognize("data/test/prapthi2.jpg", threshold=0.50, ctx_id=-1)
```

This value was selected based on the current small validation evaluation. It
is an initial working threshold, not a universally optimal value. A larger,
more representative validation dataset should be used before deploying this
system in a real application.

## Basic Evaluation

The evaluation script is [evaluation_test.py](evaluation_test.py). It compares:

- **Genuine:** `prapthi1.jpg` against `prapthi2.jpg`
- **Impostor:** `prapthi1.jpg` against `unknown.jpg`

The following thresholds were tested:

```text
0.30, 0.35, 0.40, 0.45, 0.50,
0.55, 0.60, 0.65, 0.70
```

Observed results for this evaluation:

| Threshold | Genuine accept % | Impostor accept % | Accuracy % |
|---|---:|---:|---:|
| 0.30 | 100.00 | 0.00 | 100.00 |
| 0.35 | 100.00 | 0.00 | 100.00 |
| 0.40 | 100.00 | 0.00 | 100.00 |
| 0.45 | 100.00 | 0.00 | 100.00 |
| 0.50 | 100.00 | 0.00 | 100.00 |
| 0.55 | 100.00 | 0.00 | 100.00 |
| 0.60 | 100.00 | 0.00 | 100.00 |
| 0.65 | 100.00 | 0.00 | 100.00 |
| 0.70 | 0.00 | 0.00 | 50.00 |

This is only a **basic sanity check** because the evaluation contains one
genuine pair and one impostor pair. It is too small to estimate real-world
accuracy, error rates, or generalization performance.

Run it from the project root:

```powershell
python evaluation_test.py
```

## Failure Cases

The system may fail to produce a reliable result when:

- No face is present.
- Multiple faces are present; the current workflow requires exactly one.
- Lighting is poor.
- The face is at an extreme pose or viewing angle.
- The image is blurred.
- Important facial regions are occluded.
- The input image is low quality or too small.
- The person is not enrolled, in which case the result should be `Unknown`.

## Limitations and Possible Improvements

- Evaluate on a larger and more diverse dataset.
- Calibrate the threshold using a statistically meaningful validation set.
- Support multiple enrollment images per person and aggregate their embeddings.
- Replace the JSON files with a persistent database for larger deployments.
- Add liveness detection to help reduce presentation or spoofing attacks.
- Improve handling of multiple faces, including per-face identification.

## Installation and Usage on Windows

Open PowerShell in the project directory and create a virtual environment:

```powershell
py -3.11 -m venv venv
.\venv\Scripts\Activate.ps1
```

Install the required packages:

```powershell
python -m pip install --upgrade pip
python -m pip install insightface onnxruntime opencv-python numpy
```

The first InsightFace run may download the `buffalo_l` model files. Ensure
that the machine has internet access for this initial setup.

Run the existing example scripts:

```powershell
python enroll_test.py
python recognize_test.py
python evaluation_test.py
```

The scripts use `ctx_id=-1` for CPU execution. Use `ctx_id=0` only when a
compatible GPU runtime and configuration are available.

## Project Structure

```text
face-recognition-system/
├── app/
│   ├── embedder.py       # Reusable InsightFace model and embedding extraction
│   ├── enrollment.py     # Enrollment and JSON database loading
│   ├── evaluation.py     # Pair-based threshold evaluation
│   ├── matcher.py        # Cosine similarity helper
│   └── recognition.py    # Threshold-based identity recognition
├── data/
│   ├── enrolled/         # JSON enrollment records
│   └── test/             # Evaluation and recognition images
├── enroll_test.py        # Enrollment example
├── evaluation_test.py    # Evaluation example
├── recognize_test.py     # Recognition example
├── test_face.py          # Face detection test
├── test_similarity.py    # Similarity test
└── README.md
```

## Example Recognition Output

For a successful match, the result has this form:

```python
{
    "status": "matched",
    "message": "Matched Prapthi",
    "name": "Prapthi",
    "similarity": 0.78
}
```

For a face below the configured threshold:

```python
{
    "status": "unknown",
    "message": "No enrolled identity passed the threshold",
    "similarity": 0.43
}
```

The similarity values above are representative output formats. They are not
claimed evaluation results for all images or environments.
