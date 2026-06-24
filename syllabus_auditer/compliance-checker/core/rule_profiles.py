"""
Load and save compliance rule profiles from/to JSON files in the profiles/ folder.
A profile is a named list of plain-English compliance rules.
"""

import json
from pathlib import Path

PROFILES_DIR = Path(__file__).parent.parent / "profiles"


def list_profiles() -> list[str]:
    return [p.stem for p in PROFILES_DIR.glob("*.json")]


def load_profile(name: str) -> dict:
    path = PROFILES_DIR / f"{name}.json"
    if not path.exists():
        raise FileNotFoundError(f"Profile '{name}' not found.")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_profile(name: str, description: str, rules: list[str]) -> None:
    PROFILES_DIR.mkdir(parents=True, exist_ok=True)
    path = PROFILES_DIR / f"{name}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"name": name, "description": description, "rules": rules}, f, indent=2)


def delete_profile(name: str) -> None:
    path = PROFILES_DIR / f"{name}.json"
    if path.exists():
        path.unlink()
