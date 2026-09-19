"""Simple threshold evaluation for labeled face-image pairs.

The evaluation treats a pair as:
    (image_path_1, image_path_2)

Genuine pairs show the same person. Impostor pairs show different people.
"""

from typing import Iterable, Sequence

from .embedder import Embedder
from .matcher import cosine_similarity


DEFAULT_THRESHOLDS = (
    0.30,
    0.35,
    0.40,
    0.45,
    0.50,
    0.55,
    0.60,
    0.65,
    0.70,
)


def _pair_similarities(
    pairs: Iterable[Sequence[str]],
    embedder: Embedder,
) -> list[float]:
    """Return one cosine similarity for every image pair."""
    similarities = []
    embedding_cache = {}

    for pair in pairs:
        if len(pair) != 2:
            raise ValueError("Every pair must contain exactly two image paths")

        image_path_1, image_path_2 = pair
        if image_path_1 not in embedding_cache:
            embedding_cache[image_path_1] = embedder.get_embedding_for_single_face(
                image_path_1
            )
        if image_path_2 not in embedding_cache:
            embedding_cache[image_path_2] = embedder.get_embedding_for_single_face(
                image_path_2
            )

        similarity = cosine_similarity(
            embedding_cache[image_path_1],
            embedding_cache[image_path_2],
        )
        similarities.append(float(similarity))

    return similarities


def evaluate_thresholds(
    genuine_pairs: Iterable[Sequence[str]],
    impostor_pairs: Iterable[Sequence[str]],
    thresholds: Iterable[float] = DEFAULT_THRESHOLDS,
    ctx_id: int = 0,
) -> list[dict[str, float]]:
    """Evaluate genuine and impostor image pairs at several thresholds.

    Args:
        genuine_pairs: Same-person image pairs.
        impostor_pairs: Different-person image pairs.
        thresholds: Similarity thresholds to evaluate.
        ctx_id: InsightFace context id (0 for GPU, -1 for CPU).

    Returns:
        A list of result dictionaries, one for each threshold. Rates and
        accuracy are percentages in the range 0 to 100.

    Raises:
        ValueError: If a threshold is outside [0, 1], a pair is malformed,
            or there are no evaluation pairs.
    """
    thresholds = list(thresholds)
    if any(threshold < 0 or threshold > 1 for threshold in thresholds):
        raise ValueError("Thresholds must be between 0 and 1")

    genuine_pairs = list(genuine_pairs)
    impostor_pairs = list(impostor_pairs)
    if not genuine_pairs and not impostor_pairs:
        raise ValueError("At least one genuine or impostor pair is required")

    embedder = Embedder.get_instance(ctx_id=ctx_id)
    genuine_similarities = _pair_similarities(genuine_pairs, embedder)
    impostor_similarities = _pair_similarities(impostor_pairs, embedder)

    genuine_count = len(genuine_similarities)
    impostor_count = len(impostor_similarities)
    total_count = genuine_count + impostor_count
    results = []

    for threshold in thresholds:
        genuine_accepted = sum(
            similarity >= threshold for similarity in genuine_similarities
        )
        impostor_accepted = sum(
            similarity >= threshold for similarity in impostor_similarities
        )

        genuine_accept_rate = (
            100 * genuine_accepted / genuine_count if genuine_count else 0.0
        )
        false_accept_rate = (
            100 * impostor_accepted / impostor_count if impostor_count else 0.0
        )
        correctly_classified = genuine_accepted + (impostor_count - impostor_accepted)

        results.append(
            {
                "threshold": float(threshold),
                "genuine_accept_rate": genuine_accept_rate,
                "impostor_accept_rate": false_accept_rate,
                "false_accept_rate": false_accept_rate,
                "accuracy": 100 * correctly_classified / total_count,
            }
        )

    return results


def print_results(results: Iterable[dict[str, float]]) -> None:
    """Print evaluation results as an interview-friendly table."""
    print(
        f"{'Threshold':>9} | {'Genuine accept %':>16} | "
        f"{'Impostor accept %':>17} | {'Accuracy %':>10}"
    )
    print("-" * 64)
    for result in results:
        print(
            f"{result['threshold']:>9.2f} | "
            f"{result['genuine_accept_rate']:>16.2f} | "
            f"{result['impostor_accept_rate']:>17.2f} | "
            f"{result['accuracy']:>10.2f}"
        )


if __name__ == "__main__":
    print(
        "Import evaluate_thresholds() and provide genuine_pairs and "
        "impostor_pairs image paths."
    )
