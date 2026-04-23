"""
Minimal end-to-end example demonstrating the Arca Musarithmica engine.

Run:
    python3 examples/example.py
from the project root.
"""

from __future__ import annotations

import os
import sys

# Allow running directly without install
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from arca import Arca


def main() -> None:
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "output")
    out_dir = os.path.abspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    texts = [
        "Cantate Domino canticum novum",    # Latin, paper's genre
        "Hello world",                      # English
        "안녕하세요 코워크",                    # Korean (Hangul)
    ]

    for i, text in enumerate(texts):
        print("=" * 78)
        print(f"Example {i + 1}: {text}")
        print("=" * 78)

        arca = Arca(mode="F_major", tablet_index=0)
        result = arca.compose(text)

        print(result.pretty())
        print()

        # Write outputs
        base = os.path.join(out_dir, f"example_{i + 1}")
        result.to_midi(base + ".mid")
        result.to_musicxml(base + ".musicxml")
        result.to_rule_log_json(base + ".rulelog.json")
        print(f"Wrote: {base}.mid, {base}.musicxml, {base}.rulelog.json")
        print()


if __name__ == "__main__":
    main()
