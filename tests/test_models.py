import warnings
warnings.simplefilter("ignore", DeprecationWarning)

import sys
from pathlib import Path
import asyncio
import re

root = str(Path(__file__).resolve().parents[1])
sys.path.append(root)

from finworld.models import model_manager, ChatMessage

if __name__ == "__main__":
    model_manager.init_models(use_local_proxy=False)
    
    messages = [
        {
            "role": "user",
            "content": "What is the capital of France?"
        }
    ]

    messages = [ChatMessage.from_dict(msg) for msg in messages]
    
    response = asyncio.run(model_manager.registed_models["Qwen3-1.7B"](
        messages=messages,
    ))
    print(response)
    cleaned = re.sub(r"<think>.*?</think>\s*", "", response.content, flags=re.S).strip()
    print(cleaned)
