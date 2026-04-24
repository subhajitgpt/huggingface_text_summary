from hf_text_summary.analysis import (
    choose_summary_point_count,
    extract_key_phrases,
    generate_summary_points,
)


def test_extract_key_phrases_dedup_and_limit():
    text = "Streamlit app. Streamlit app. Hugging Face models for summarization and intent."
    phrases = extract_key_phrases(text, top_k=5)
    assert len(phrases) <= 5
    assert len({p.lower() for p in phrases}) == len(phrases)


def test_choose_summary_point_count_scales_with_length():
    short = "word " * 100
    medium = "word " * 600
    long = "word " * 2000

    assert choose_summary_point_count(short) == 5
    assert 5 <= choose_summary_point_count(medium) <= 10
    assert choose_summary_point_count(long) == 10


def test_generate_summary_points_respects_target_cap():
    # 30 short sentences -> enough material for the 5..10 range.
    text = " ".join([f"Sentence {i} mentions performance and stability." for i in range(1, 31)])
    points, meta = generate_summary_points(text, min_points=5, max_points=10)

    assert meta["mode"] == "extractive"
    assert 1 <= len(points) <= 10
    # For this input length we should land within the configured cap.
    assert len(points) <= meta["target_points"]
