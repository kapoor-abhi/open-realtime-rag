#vision.py
"""
FIXED: Removed @observe decorators from functions called inside a
ThreadPoolExecutor. The OTel ContextVar token created in the thread pool
worker cannot be detached in the parent context, producing:

  ValueError: <Token ...> was created in a different Context

The Groq LLM generation is already traced at the graph level via
CallbackHandler in generate_node, so nothing is lost observability-wise.
If you need per-image tracing, call generate_image_caption from an async
context (not a thread) and re-add @observe there.
"""

import base64
from groq import Groq
from langfuse import get_client
from app.core.config import get_settings


# No @observe here — this runs inside a ThreadPoolExecutor and OTel
# ContextVar tokens must not cross thread boundaries.
def encode_image(image_path: str) -> str:
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def generate_image_caption(image_path: str) -> str:
    """
    Generate a caption for an image using Groq Vision.

    NOTE: @observe intentionally removed. This function is called from
    parser.py's ThreadPoolExecutor. OpenTelemetry ContextVar tokens
    cannot be safely shared across threads, causing:

        ValueError: <Token ...> was created in a different Context

    Langfuse tracing for the overall document-processing pipeline is
    handled at the worker / graph level.
    """
    settings = get_settings()
    client = Groq(api_key=settings.GROQ_API_KEY)
    model_name = "meta-llama/llama-4-scout-17b-16e-instruct"

    base64_image = encode_image(image_path)

    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "Analyze this image in detail. "
                            "If it is a chart, graph, or table, extract the key data points and trends. "
                            "If it is a diagram, explain the flow or structure."
                        ),
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{base64_image}",
                        },
                    },
                ],
            }
        ],
        model=model_name,
    )

    content = chat_completion.choices[0].message.content
    return content