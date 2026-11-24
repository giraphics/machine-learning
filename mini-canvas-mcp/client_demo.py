"""
client_demo.py - an MCP client (a "host") for the Mini Canvas server.

Two modes, so students can compare them side by side:

  1. HARD-CODED (no AI)     python client_demo.py
     We write the plan ourselves: which tools to call, with which arguments.

  2. LLM (an AI decides)    python client_demo.py --llm "a snowman at night"
     We hand the server's tool list to an LLM. The LLM decides which tools
     to call; this script runs them on the MCP server and sends the results
     back, in a loop, until the LLM says it is done.

     Default LLM is a local Ollama model (picked automatically from what is
     installed). Other options:
        python client_demo.py --llm "a house" --provider claude
        python client_demo.py --llm "a house" --provider openai --model gpt-5.5
        python client_demo.py --llm "a house" --provider ollama --model llama3.1

Both modes use the SAME MCP steps:
   initialize -> tools/list -> tools/call ... -> resources/read
The only difference is WHO chooses the tool calls.
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path

from mcp import Client, StdioServerParameters

from llm import LLMError, create_chat

SERVER = StdioServerParameters(
    command=sys.executable,  # the same Python running this script
    args=[str(Path(__file__).parent / "server.py")],
)

MAX_STEPS = 30  # safety limit on LLM <-> tool round trips


def step(title: str) -> None:
    print(f"\n=== {title} " + "=" * max(3, 60 - len(title)))


def result_text(result) -> str:
    return "\n".join(c.text for c in result.content if hasattr(c, "text"))


async def show_server(client: Client) -> list:
    """tools/list, resources/list and prompts/list - what the server offers."""
    step("What does the server offer? (tools/list)")
    tools = (await client.list_tools()).tools
    for tool in tools:
        params = ", ".join(tool.input_schema.get("properties", {}).keys())
        first_line = (tool.description or "").strip().splitlines()[0]
        print(f"- {tool.name}({params})\n    {first_line}")
    for r in (await client.list_resources()).resources:
        print(f"- resource: {r.uri}")
    for p in (await client.list_prompts()).prompts:
        print(f"- prompt:   {p.name}")
    return tools


# ============================================================================
# Mode 1 - hard-coded plan (no AI)
# ============================================================================

async def run_hardcoded(client: Client) -> None:
    step("Call tools - WE wrote this plan by hand (tools/call)")
    plan = [
        ("new_canvas",  {"width": 400, "height": 300, "background": "#87ceeb"}),
        ("draw_rect",   {"x": 0, "y": 200, "width": 400, "height": 100, "colour": "#3cb371"}),
        ("draw_circle", {"x": 320, "y": 70, "radius": 40, "colour": "gold"}),
        ("draw_rect",   {"x": 90, "y": 130, "width": 120, "height": 90, "colour": "#cd853f"}),
        ("draw_line",   {"x1": 80, "y1": 135, "x2": 150, "y2": 80, "colour": "brown", "thickness": 6}),
        ("draw_line",   {"x1": 150, "y1": 80, "x2": 220, "y2": 135, "colour": "brown", "thickness": 6}),
        ("draw_text",   {"x": 20, "y": 30, "text": "Hello MCP!", "size": 24}),
        ("save_svg",    {"filename": "client_demo.svg"}),
    ]
    for name, args in plan:
        result = await client.call_tool(name, args)
        print(f"-> {name}({args})\n   <- {result_text(result)}")


# ============================================================================
# Mode 2 - an LLM chooses the tool calls
# ============================================================================

def mcp_tools_for_llm(tools) -> list[dict]:
    """MCP tool list -> the plain format llm.py expects."""
    return [{"name": t.name, "description": t.description or "", "parameters": t.input_schema}
            for t in tools]


async def run_llm(client: Client, tools, subject: str, provider: str | None,
                  model: str | None, filename: str) -> None:
    chat = create_chat(provider, model, system=(
        "You are a careful illustrator. You can ONLY draw by calling the provided tools. "
        "Plan coordinates so every shape is inside the canvas."
    ))
    step(f"LLM mode - using {chat}")

    # Use the server's own PROMPT template to build the request (prompts/get)
    prompt = await client.get_prompt("draw_scene", {"subject": subject})
    request = prompt.messages[0].content.text
    request += f'\nSave it with the filename "{filename}".'
    print("User request (from the server's draw_scene prompt):")
    print("  " + request.replace("\n", "\n  "))
    chat.add_user(request)

    llm_tools = mcp_tools_for_llm(tools)
    for n in range(1, MAX_STEPS + 1):
        reply = chat.send(llm_tools)                      # 1. LLM thinks
        if reply.text.strip():
            print(f"\n[LLM says] {reply.text.strip()}")
        if not reply.tool_calls:                          # no tool calls -> finished
            break
        for call in reply.tool_calls:                     # 2. LLM asked for tools
            print(f"-> step {n}: {call.name}({json.dumps(call.arguments)})")
            try:
                result = await client.call_tool(call.name, call.arguments)  # 3. MCP runs it
                text = result_text(result)
            except Exception as e:  # e.g. bad arguments - tell the LLM so it can fix it
                text = f"Error: {e}"
            print(f"   <- {text}")
            chat.add_tool_result(call, text)              # 4. result goes back to the LLM
    else:
        print(f"\nStopped after {MAX_STEPS} steps (safety limit).")


# ============================================================================

async def main() -> None:
    ap = argparse.ArgumentParser(description="Mini Canvas MCP client demo")
    ap.add_argument("--llm", metavar="SUBJECT",
                    help='let an LLM draw this, e.g. --llm "a snowman at night"')
    ap.add_argument("--provider", choices=["ollama", "claude", "openai"],
                    help="LLM provider (default: ollama, or LLM_PROVIDER)")
    ap.add_argument("--model", help="model name (default: auto-picked for Ollama)")
    ap.add_argument("--out", default="llm_drawing.svg", help="file name for LLM mode")
    args = ap.parse_args()

    async with Client(SERVER) as client:
        tools = await show_server(client)

        if args.llm:
            try:
                await run_llm(client, tools, args.llm, args.provider, args.model, args.out)
            except LLMError as e:
                print(f"\nLLM error: {e}")
                return
        else:
            await run_hardcoded(client)

        step("Read a resource (resources/read)")
        svg = (await client.read_resource("canvas://current")).contents[0].text
        print(svg[:300] + ("..." if len(svg) > 300 else ""))

    print("\nDone. Open the .svg file in the output folder in a browser to see the picture.")


if __name__ == "__main__":
    asyncio.run(main())
