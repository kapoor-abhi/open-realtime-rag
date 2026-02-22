#vision.py
import base64
from groq import Groq
from app.core.config import get_settings

def encode_image(image_path: str) -> str:
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def generate_image_caption(image_path: str) -> str:
    settings = get_settings()
    client = Groq(api_key=settings.GROQ_API_KEY)
    
    try:
        base64_image = encode_image(image_path)
        
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
            model="meta-llama/llama-4-scout-17b-16e-instruct",
        )
        return chat_completion.choices[0].message.content
    except Exception:
        pass
        
    return "An image found in the document."