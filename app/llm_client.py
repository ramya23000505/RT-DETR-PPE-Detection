import os
import logging
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from google import genai
from google.genai.errors import APIError

logger = logging.getLogger("rt_detr_ppe.llm")

def _exec_gemini(prompt: str, system_instruction: str) -> str:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return "Gemini API Key missing in environment."
    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model='gemini-2.5-flash',
        contents=prompt,
        config=genai.types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=0.1,
            max_output_tokens=500
        )
    )
    if response and response.text:
        return response.text.strip()
    return "Unable to produce reasoning response from LLM."

def call_gemini_reasoning(prompt: str, system_instruction: str, timeout_sec: int = 15) -> str:
    """
    Isolated Google Gemini API client with strict ThreadPoolExecutor timeout and error handling.
    """
    if not os.getenv("GEMINI_API_KEY"):
        logger.warning("GEMINI_API_KEY not found in environment variables.")
        return "Gemini API Key missing in environment."
        
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(_exec_gemini, prompt, system_instruction)
        try:
            return future.result(timeout=timeout_sec)
        except FutureTimeoutError:
            logger.error(f"Gemini API request timed out after {timeout_sec} seconds.")
            return f"LLM Service Timeout: Request exceeded {timeout_sec}s threshold."
        except APIError as e:
            logger.error(f"Gemini API Error: {e}")
            return f"LLM Service Error: {e.message if hasattr(e, 'message') else str(e)}"
        except Exception as e:
            logger.error(f"Unexpected error calling Gemini: {e}")
            return f"LLM Service Unavailable: {str(e)}"
