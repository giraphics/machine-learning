"""
server.py - the MCP layer. It turns the Painter class into AI tools.

All the drawing logic lives in painter.py (plain, reusable Python).
This file only:
  1. creates one Painter,
  2. wraps its methods as MCP tools, a resource and a prompt,
  3. writes descriptions the AI reads, and checks the AI's inputs.

    painter.py  = WHAT can be done   (your existing code)
    server.py   = HOW an AI reaches it (the MCP "adapter")

Run it:      python server.py         (talks MCP over stdin/stdout)
Inspect it:  mcp dev server.py        (opens MCP Inspector in the browser)

Teaching notes:
  1. Each tool's docstring is what the AI reads to decide when to use it.
  2. Type hints (int, str) become the tool's input schema automatically.
  3. NEVER print() to stdout in a stdio server - stdout carries the MCP
     messages. Log to stderr instead (see log() below).
"""

import sys
from pathlib import Path

from mcp.server.mcpserver import MCPServer  # MCP Python SDK v2 (v1 called this FastMCP)

from painter import Painter  # <- the existing, reusable code

mcp = MCPServer("mini-canvas")
painter = Painter()
OUTPUT_DIR = Path(__file__).parent / "output"


def log(message: str) -> None:
    """Log to stderr so it never corrupts the MCP messages on stdout."""
    print(f"[mini-canvas] {message}", file=sys.stderr, flush=True)


def done(what: str) -> str:
    """A short reply for the AI, so it knows the call worked."""
    return f"{what} The canvas now has {painter.shape_count} shapes."


# ============================================================================
# TOOLS - each one is a thin wrapper around a Painter method
# ============================================================================

@mcp.tool()
def new_canvas(width: int = 400, height: int = 400, background: str = "white") -> str:
    """Start a new blank canvas. Call this first before drawing.
    (0, 0) is the TOP-LEFT corner; x grows to the right, y grows downward.
    Colours can be names ("skyblue") or hex codes ("#ff8800")."""
    painter.new_canvas(width, height, background)
    log(f"new_canvas {width}x{height} {background}")
    return f"New {width}x{height} canvas with a {background} background."


@mcp.tool()
def draw_circle(x: int, y: int, radius: int, colour: str) -> str:
    """Draw a filled circle centred at (x, y)."""
    painter.circle(x, y, radius, colour)
    log(f"draw_circle ({x},{y}) r={radius} {colour}")
    return done("Circle drawn.")


@mcp.tool()
def draw_rect(x: int, y: int, width: int, height: int, colour: str) -> str:
    """Draw a filled rectangle. (x, y) is its TOP-LEFT corner."""
    painter.rect(x, y, width, height, colour)
    log(f"draw_rect ({x},{y}) {width}x{height} {colour}")
    return done("Rectangle drawn.")


@mcp.tool()
def draw_line(x1: int, y1: int, x2: int, y2: int, colour: str, thickness: int = 2) -> str:
    """Draw a straight line from (x1, y1) to (x2, y2)."""
    painter.line(x1, y1, x2, y2, colour, thickness)
    log(f"draw_line ({x1},{y1})->({x2},{y2}) {colour}")
    return done("Line drawn.")


@mcp.tool()
def draw_text(x: int, y: int, text: str, colour: str = "black", size: int = 20) -> str:
    """Write text on the canvas. (x, y) is where the text's baseline starts."""
    painter.text(x, y, text, colour, size)
    log(f"draw_text ({x},{y}) {text!r}")
    return done("Text drawn.")


@mcp.tool()
def undo() -> str:
    """Remove the most recently drawn shape."""
    if not painter.undo():
        return "Nothing to undo - the canvas is empty."
    log("undo")
    return done("Removed the last shape.")


@mcp.tool()
def save_svg(filename: str = "drawing.svg") -> str:
    """Save the current canvas as an SVG image file in the output folder.
    Call this when the drawing is finished."""
    # Security belongs HERE, at the boundary where AI input arrives:
    # keep only the bare file name so the AI can't write anywhere else
    # on the computer (e.g. "../../important.txt").
    safe_name = Path(filename).name or "drawing.svg"
    if not safe_name.endswith(".svg"):
        safe_name += ".svg"
    path = painter.save(OUTPUT_DIR / safe_name)
    log(f"save_svg -> {path}")
    return f"Saved {painter.shape_count} shapes to {path}"


# ============================================================================
# RESOURCES - data the AI (or the user) can READ
# ============================================================================

@mcp.resource("canvas://current", mime_type="image/svg+xml")
def current_canvas() -> str:
    """The current drawing as SVG text."""
    return painter.to_svg()


# ============================================================================
# PROMPTS - reusable templates the user can pick in the AI app
# ============================================================================

@mcp.prompt()
def draw_scene(subject: str) -> str:
    """Ask the AI to plan and draw a whole scene."""
    return (
        f"Using the mini-canvas tools, draw: {subject}.\n"
        "First, briefly plan the shapes and their coordinates. "
        "Then start a new canvas, draw from the background to the foreground, "
        "and finish by saving the SVG."
    )


if __name__ == "__main__":
    log("server starting (stdio)")
    mcp.run()  # default transport is stdio
