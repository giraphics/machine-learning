# Mini Canvas – a tiny MCP server for teaching

A small **Model Context Protocol (MCP)** server that gives an AI a set of drawing tools.
The AI can't draw on its own. When you ask Claude to *"draw a house at sunset"*, it calls
this server's tools one at a time (`new_canvas` → `draw_rect` → `draw_circle` → … → `save_svg`),
and the server writes the SVG image.

```
 ┌──────────────── HOST (Claude Desktop) ────────────────┐
 │   You: "draw a house at sunset"                       │
 │   Model: decides which tools to call  ──┐             │
 │   MCP Client ◄──────────────────────────┘             │
 └──────┬────────────────────────────────────────────────┘
        │  JSON-RPC messages over stdin/stdout
 ┌──────▼──────────── MCP SERVER (server.py) ────────────┐
 │   Tools:     new_canvas, draw_circle, draw_rect,      │
 │              draw_line, draw_text, undo, save_svg     │
 │   Resource:  canvas://current   (the SVG so far)      │
 │   Prompt:    draw_scene(subject)                      │
 │        thin wrapper: each tool calls one method ▼     │
 ├──────────────── EXISTING CODE (painter.py) ───────────┤
 │   class Painter: circle(), rect(), line(), text(),    │
 │                  undo(), to_svg(), save()             │
 │   plain Python - knows nothing about MCP or AI        │
 └───────────────────────────────────────────────────────┘
                     │
                     ▼  output/drawing.svg
```

## Files

| File | What it is |
|---|---|
| `painter.py` | **Reusable drawing library.** A plain Python `Painter` class with no MCP or AI in it. You could use it in any program. |
| `server.py` | **The MCP layer.** It wraps `Painter` methods as tools, a resource and a prompt for the AI. |
| `llm.py` | **Reusable LLM connector.** One interface for Ollama (the default, local), Claude and OpenAI, including tool calling. It knows nothing about MCP, so you can reuse it in other projects. |
| `client_demo.py` | The MCP client (host). It runs in two modes: a **hard-coded** plan, or an **LLM** that chooses the tool calls. |
| `output/` | Saved SVG images (created automatically). |

**The main message:** you don't rewrite your code for AI. `painter.py` is the code you
already have, and `server.py` is a thin adapter that lets an AI call it.
The same idea works for any existing library: a physics engine, a database, a model you trained.

## 1. Setup (Windows)

Requires Python 3.10 or newer.

```powershell
cd C:\dev\giraphics\machine-learning\mini-canvas-mcp
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

> This uses **MCP Python SDK v2**, where the server class is `MCPServer`
> (`from mcp.server.mcpserver import MCPServer`). Older tutorials use v1's
> `FastMCP`. The idea is the same; only the name changed.

## 2. Demo 0 – the Painter library alone (no MCP, no AI)

```powershell
python painter.py
```

A human writes the drawing steps in Python and gets `output/painter_demo.svg`.
Show students that this is ordinary code. Then ask: *"How could an AI use this?"*

```python
from painter import Painter
p = Painter(400, 300, "skyblue")
p.circle(320, 70, 40, "gold")
p.save("output/sun.svg")
```

## 3. Demo A – the protocol with no AI

```powershell
python client_demo.py
```

The script goes through the same steps an AI host goes through:

1. **connect** (initialize handshake)
2. **`tools/list`**: "what can you do?" Each tool comes back with its name, description and parameters.
3. **`tools/call`**: run tools with arguments.
4. **`resources/read`**: read data.

Open `output/client_demo.svg` in a browser to see the result.
**Key point:** an AI does exactly this. The difference is that the *model chooses* the calls.

## 4. Demo A2 – the same client, but an LLM decides

```powershell
python client_demo.py --llm "a snowman on a starry night"
```

The client sends the server's tool list to an LLM, then runs this loop:

```
 LLM: "call draw_circle(x=200, y=250, ...)"  ──►  client ──► MCP server (tools/call)
 LLM ◄── "Circle drawn. 3 shapes."           ◄──  client ◄── result
 ... repeats until the LLM stops asking for tools
```

The MCP steps are the same as in Demo A. **Only who chooses the calls has changed.**
The request is built from the server's own `draw_scene` prompt (`prompts/get`), so the demo uses tools, resources and prompts.

**Ollama (default, free, runs locally)**

1. Install Ollama from https://ollama.com.
2. Pull a model that supports tool calling, for example `ollama pull qwen3:8b`. On a laptop with less memory, use `ollama pull llama3.2:3b`.
3. Check what `llm.py` sees on your PC:
   ```powershell
   python llm.py
   ```
   This lists your installed models, marks which ones support tools, and picks one automatically.
   The order it tries is `OLLAMA_PREFERENCE` in `llm.py`: qwen3, gpt-oss, llama3.1, ... It takes the smallest size of the first family found.

**Claude or OpenAI (need an API key)**

```powershell
$env:ANTHROPIC_API_KEY = "sk-ant-..."
python client_demo.py --llm "a lighthouse at night" --provider claude

$env:OPENAI_API_KEY = "sk-..."
python client_demo.py --llm "a lighthouse at night" --provider openai
```

**Choose a specific model:** `--model llama3.1:8b`, or set `$env:LLM_PROVIDER` / `$env:LLM_MODEL` once.

**Classroom idea:** run the same subject with a small local model and with Claude, and compare the pictures and the tool calls.
The tools and server don't change. Only the "brain" does.

### Using `llm.py` in your own code

```python
from llm import create_chat

chat = create_chat()                      # Ollama, auto-picked model
print(chat.ask("Explain MCP in one sentence"))

tools = [{"name": "add", "description": "Add two numbers",
          "parameters": {"type": "object",
                         "properties": {"a": {"type": "number"}, "b": {"type": "number"}},
                         "required": ["a", "b"]}}]

chat = create_chat("claude", system="Be brief.")
chat.add_user("What is 2 + 2? Use the add tool.")
reply = chat.send(tools)
for call in reply.tool_calls:
    chat.add_tool_result(call, str(call.arguments["a"] + call.arguments["b"]))
print(chat.send(tools).text)              # always pass the same tools again
```

Each provider uses a different JSON format: Ollama `/api/chat`, Anthropic `/v1/messages` and OpenAI `/v1/responses`.
`llm.py` hides those differences. That also makes a good comparison for students: MCP is the standard on the *tool* side, while LLM APIs are still all different.

## 5. Demo B – inspect it with MCP Inspector

Needs Node.js (`npx`).

```powershell
mcp dev server.py
```

or

```powershell
npx @modelcontextprotocol/inspector python server.py
```

In the browser UI, open **Tools → List Tools**, pick `draw_circle`, fill in the arguments and run it.
Students can see the raw JSON requests and responses.

## 6. Demo C – connect it to Claude Desktop

Open **Claude Desktop → Settings → Developer → Edit Config**. This opens the config file Claude Desktop actually uses.
Its location varies between installs. It is often `%APPDATA%\Claude\claude_desktop_config.json`, but on some installs it is
`%LOCALAPPDATA%\Claude-Data\<id>\claude_desktop_config.json`. To find it, run:
`Get-ChildItem "$env:USERPROFILE\AppData" -Recurse -Filter claude_desktop_config.json -ErrorAction SilentlyContinue`

**Quit Claude Desktop first** (tray icon → Quit), because it rewrites this file while it runs. Add `mcpServers` as a top-level key, next to `"preferences"`:

```json
{
  "mcpServers": {
    "mini-canvas": {
      "command": "C:\\dev\\giraphics\\machine-learning\\mini-canvas-mcp\\.venv\\Scripts\\python.exe",
      "args": ["C:\\dev\\giraphics\\machine-learning\\mini-canvas-mcp\\server.py"]
    }
  }
}
```

Fully quit and restart Claude Desktop, then try:

- *"Using mini-canvas, draw a snowman on a blue night sky and save it as snowman.svg"*
- *"Draw a traffic light, then undo the last shape and make the green light bigger"*
- Choose the **draw_scene** prompt from the + menu and enter a subject.

Claude shows each tool call it makes. Expand them in class so students can see the arguments it chose.

## Teaching points in the code

1. **The docstring is the interface.** The model reads each tool's description to decide when and how to use it.
   *Live experiment:* change `draw_rect`'s docstring to remove "(x, y) is its TOP-LEFT corner", restart, and watch placement get worse.
2. **Type hints become the schema.** `x: int` is what tells the model that `x` must be a number.
3. **stdout is reserved.** In a stdio server, `print()` to stdout would corrupt the protocol, so `log()` writes to stderr.
4. **Servers run real code, so be careful.** `save_svg` strips paths so the AI can't write outside `output/`. `escape()` in the Painter stops it from injecting markup.
5. **The model plans and the server executes.** The server has no intelligence of its own. It only does what it's asked.
6. **Keep the layers separate.** `painter.py` handles correct drawing (for example, escaping text for SVG).
   `server.py` handles AI-facing concerns: descriptions, friendly replies and input safety (the `save_svg` filename check).

## Student exercises

| Level | Task |
|---|---|
| Easy | Add `ellipse()` to `Painter`, then expose it as a `draw_ellipse` tool. Notice it's a two-step change: library first, then the MCP wrapper. |
| Easy | Add a `clear()` tool that empties the canvas but keeps the size. |
| Medium | Add `draw_polygon(points: list[list[int]], colour)` so roofs can be real triangles. |
| Medium | Add an optional `opacity: float = 1.0` to every shape. |
| Medium | Add a resource `canvas://shapes` that returns the shape list as JSON. |
| Challenge | Add `transform_last(dx, dy, scale)`: translation and scaling, the basics of graphics transforms. |
| Medium | Add a fourth provider to `llm.py`, e.g. any OpenAI-compatible local server such as LM Studio. |
| Challenge | Support several named layers (`background`, `foreground`) so the AI can redraw one layer. |
