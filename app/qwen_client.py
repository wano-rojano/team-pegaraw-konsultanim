import os
import dashscope
from dotenv import load_dotenv

load_dotenv()

dashscope.base_http_api_url = "https://dashscope-intl.aliyuncs.com/api/v1"


def qwen_chat(messages, model: str = "qwen3-max", result_format: str = "message") -> str:
    """Call Qwen via DashScope and return the text content."""
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        raise RuntimeError("DASHSCOPE_API_KEY is not set")

    resp = dashscope.Generation.call(
        api_key=api_key,
        model=model,
        messages=messages,
        result_format=result_format,
    )
    # Basic error handling
    if getattr(resp, "status_code", 200) != 200:
        raise RuntimeError(
            f"DashScope error: status_code={getattr(resp, 'status_code', None)}, "
            f"code={getattr(resp, 'code', None)}, message={getattr(resp, 'message', resp)}"
        )

    return resp.output.choices[0].message.content
