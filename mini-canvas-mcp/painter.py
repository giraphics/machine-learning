"""
painter.py - a small, ordinary Python drawing library.

There is NOTHING about MCP or AI in this file. It is just a reusable class
you could use in any program, a notebook, a web app or a unit test.

The point for students: you usually don't write new code for MCP. You take
code you ALREADY have (like this Painter) and put a thin MCP layer on top
(see server.py), so an AI can use it too.

Try it on its own:
    python painter.py         -> writes output/painter_demo.svg
"""

from html import escape
from pathlib import Path


class Painter:
    """Builds a picture out of simple shapes and exports it as SVG.

    Coordinates: (0, 0) is the TOP-LEFT corner; x grows to the right,
    y grows downward (the usual convention in screen graphics).
    Colours: any SVG/CSS colour, e.g. "skyblue" or "#ff8800".
    """

    def __init__(self, width: int = 400, height: int = 400, background: str = "white"):
        self.new_canvas(width, height, background)

    # ---- canvas -----------------------------------------------------------
    def new_canvas(self, width: int = 400, height: int = 400, background: str = "white") -> None:
        """Clear everything and start a blank canvas."""
        self.width = width
        self.height = height
        self.background = background
        self.shapes: list[str] = []  # each item is one SVG element

    # ---- shapes -----------------------------------------------------------
    def circle(self, x: int, y: int, radius: int, colour: str) -> None:
        self.shapes.append(f'<circle cx="{x}" cy="{y}" r="{radius}" fill="{escape(colour)}"/>')

    def rect(self, x: int, y: int, width: int, height: int, colour: str) -> None:
        self.shapes.append(
            f'<rect x="{x}" y="{y}" width="{width}" height="{height}" fill="{escape(colour)}"/>'
        )

    def line(self, x1: int, y1: int, x2: int, y2: int, colour: str, thickness: int = 2) -> None:
        self.shapes.append(
            f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" '
            f'stroke="{escape(colour)}" stroke-width="{thickness}" stroke-linecap="round"/>'
        )

    def text(self, x: int, y: int, text: str, colour: str = "black", size: int = 20) -> None:
        self.shapes.append(
            f'<text x="{x}" y="{y}" font-size="{size}" font-family="sans-serif" '
            f'fill="{escape(colour)}">{escape(text)}</text>'
        )

    # ---- editing ----------------------------------------------------------
    def undo(self) -> bool:
        """Remove the last shape. Returns False if there was nothing to remove."""
        if not self.shapes:
            return False
        self.shapes.pop()
        return True

    @property
    def shape_count(self) -> int:
        return len(self.shapes)

    # ---- output -----------------------------------------------------------
    def to_svg(self) -> str:
        """Return the whole picture as an SVG document (a string)."""
        body = "\n  ".join(self.shapes)
        return (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.width}" '
            f'height="{self.height}" viewBox="0 0 {self.width} {self.height}">\n'
            f'  <rect width="100%" height="100%" fill="{escape(self.background)}"/>\n'
            f"  {body}\n"
            f"</svg>\n"
        )

    def save(self, path: str | Path) -> Path:
        """Write the SVG to a file and return its path."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_svg(), encoding="utf-8")
        return path


# Using the library directly - no AI, no MCP. A human writes the "plan".
if __name__ == "__main__":
    p = Painter(400, 300, "#87ceeb")
    p.rect(0, 200, 400, 100, "#3cb371")          # grass
    p.circle(320, 70, 40, "gold")                # sun
    p.rect(90, 130, 120, 90, "#cd853f")          # house
    p.line(80, 135, 150, 80, "brown", 6)         # roof
    p.line(150, 80, 220, 135, "brown", 6)
    p.text(20, 30, "Drawn by plain Python", size=20)
    out = p.save(Path(__file__).parent / "output" / "painter_demo.svg")
    print(f"Saved {p.shape_count} shapes to {out}")
