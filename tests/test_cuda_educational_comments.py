from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = (ROOT / "csrc/fused_causal_softmax.cu").read_text()


def test_cuda_comments_introduce_required_learning_vocabulary():
    concepts = [
        "CUDA kernel",
        "global thread index",
        "grid",
        "block",
        "warp",
        "lane",
        "shared-memory",
        "synchronization",
        "reduction",
        "shuffle",
        "coalescing",
        "numerical",
    ]
    source_lower = SOURCE.lower()
    assert all(concept.lower() in source_lower for concept in concepts)


def test_cuda_comments_explain_reasoning_not_only_mechanics():
    reasons = [
        "avoids allocating a separate",
        "identity value",
        "preventing overflow",
        "prevents a fast warp",
        "other blocks\n  // cannot",
        "rather than storage for the whole row",
    ]
    assert all(reason in SOURCE for reason in reasons)
