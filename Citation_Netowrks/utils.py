import os


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def save_text(path: str, text: str):
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def relative_error(target, generated) -> float:
    """|generated - target| / max(|target|, eps). Used for unbounded counts."""
    eps = 1e-12
    denom = max(abs(float(target)), eps)
    return abs(float(generated) - float(target)) / denom