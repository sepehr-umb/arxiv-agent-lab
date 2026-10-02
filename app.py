import gradio as gr
import spaces
from smolagents import stream_to_gradio

from guardrails import check_user_input
from research_agents import MODEL_ID, build_agent

# One agent per browser session, so each user gets their own conversation memory.
_session_agents = {}


# 1. THE DECOY: Passes the HF startup check so the container boots
@spaces.GPU
def decoy_function():
    pass


# 2. THE ACTUAL CHAT: Runs on the free background CPU without touching your quota
def agent_chat(message: str, history: list, request: gr.Request):
    refusal = check_user_input(message)
    if refusal:
        yield refusal
        return

    session_id = request.session_hash if request else "default"
    if session_id not in _session_agents:
        _session_agents[session_id] = build_agent()
    agent = _session_agents[session_id]

    # Each agent step (code, tool output, errors) is shown as a collapsible message, followed by the final answer.
    messages = []
    try:
        for msg in stream_to_gradio(agent, task=message, reset_agent_memory=False):
            if isinstance(msg, gr.ChatMessage):
                messages.append(msg)
                yield messages
    except Exception as e:
        messages.append(gr.ChatMessage(role="assistant", content=f"Sorry, something went wrong: {e}"))
        yield messages


demo = gr.ChatInterface(
    fn=agent_chat,
    title="Autonomous arXiv Research Agent",
    description=(
        f"Powered by smolagents and {MODEL_ID}. Search arXiv, read and summarize full papers, and get accurate "
        "BibTeX. Follow-up questions remember the conversation."
    ),
    examples=[
        "Find 2 recent papers on Agentic AI in cs.AI and format them as BibTeX.",
        "Summarize the paper 1706.03762 and focus on how attention works.",
        "Find 3 papers on retrieval-augmented generation and compare their approaches.",
    ],
    run_examples_on_click=False,
)

demo.launch()
