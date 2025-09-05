from pathlib import Path

def find_repo_root(start: Path | None = None) -> Path:
    """
    Walk upward from `start` (or this file) until we find a directory that
    looks like the repo root (contains 'src' or '.git'). Falls back to parent(2).
    """
    here = (start or Path(__file__).resolve()).parent
    for p in [here, *here.parents]:
        if (p / "src").is_dir() or (p / ".git").exists():
            return p
    # Fallback for layouts like <repo>/src/utils/...
    return Path(__file__).resolve().parents[2]
