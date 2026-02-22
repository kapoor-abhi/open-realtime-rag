import base64
from groq import Groq
from langfuse.decorators import observe, langfuse_context
from app.core.config import get_settings

@observe(name="encode_base64_image")
def encode_image(image_path: str) -> str:
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# as_type="generation" tells Langfuse to treat this specifically as an LLM call
@observe(as_type="generation", name="groq_vision_captioning")
def generate_image_caption(image_path: str) -> str:
    settings = get_settings()
    client = Groq(api_key=settings.GROQ_API_KEY)
    model_name = "meta-llama/llama-4-scout-17b-16e-instruct"
    
    try:
        base64_image = encode_image(image_path)
        
        # 1. Log the input and model name to Langfuse before the call
        langfuse_context.update_current_observation(
            input=f"Analyzing extracted image: {image_path}",
            model=model_name
        )
        
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text", 
                            "text": "Analyze this image in detail. If it is a chart, graph, or table, extract the key data points and trends. If it is a diagram, explain the flow or structure."
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
        
        # 2. Extract token usage from the Groq response and log it to Langfuse
        if chat_completion.usage:
            langfuse_context.update_current_observation(
                output=content,
                usage={
                    "input": chat_completion.usage.prompt_tokens,
                    "output": chat_completion.usage.completion_tokens,
                    "total": chat_completion.usage.total_tokens
                }
            )
        else:
            langfuse_context.update_current_observation(output=content)
            
        return content
        
    except Exception as e:
        # 3. If the API fails, flag the trace in Langfuse as an error with the exact message
        langfuse_context.update_current_observation(
            level="ERROR", 
            status_message=str(e)
        )
        raise e  # Reraise so our parser.py ThreadPool logger can catch it!