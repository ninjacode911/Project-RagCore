import asyncio
import logging
import time
from collections.abc import AsyncGenerator

import google.generativeai as genai

from app.config import get_settings

logger = logging.getLogger(__name__)


class GeminiService:
    def __init__(self, api_key: str, model_name: str, rpm_limit: int = 15):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)
        self.model_name = model_name
        self._min_interval = 60.0 / rpm_limit
        self._last_call_time = 0.0
        logger.info(f"Initialized Gemini '{model_name}' (RPM limit: {rpm_limit})")

    def _wait_for_rate_limit(self) -> None:
        now = time.time()
        elapsed = now - self._last_call_time
        if elapsed < self._min_interval:
            wait = self._min_interval - elapsed
            logger.debug(f"Rate limiting: waiting {wait:.1f}s")
            time.sleep(wait)
        self._last_call_time = time.time()

    async def _async_wait_for_rate_limit(self) -> None:
        now = time.time()
        elapsed = now - self._last_call_time
        if elapsed < self._min_interval:
            wait = self._min_interval - elapsed
            logger.debug(f"Rate limiting: waiting {wait:.1f}s")
            await asyncio.sleep(wait)
        self._last_call_time = time.time()

    def generate(self, prompt: str, temperature: float = 0.3, max_tokens: int = 2048) -> str:
        self._wait_for_rate_limit()
        try:
            response = self.model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=temperature,
                    max_output_tokens=max_tokens,
                ),
            )
            return response.text
        except Exception as e:
            logger.error(f"Gemini generation failed: {e}")
            raise

    async def generate_stream(
        self, prompt: str, temperature: float = 0.3, max_tokens: int = 2048
    ) -> AsyncGenerator[str, None]:
        await self._async_wait_for_rate_limit()
        try:
            response = self.model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=temperature,
                    max_output_tokens=max_tokens,
                ),
                stream=True,
            )
            for chunk in response:
                if chunk.text:
                    yield chunk.text
        except Exception as e:
            logger.error(f"Gemini streaming failed: {e}")
            raise


_llm: GeminiService | None = None


def get_llm() -> GeminiService:
    global _llm
    if _llm is None:
        settings = get_settings()
        _llm = GeminiService(
            api_key=settings.gemini_api_key,
            model_name=settings.gemini_model,
            rpm_limit=settings.gemini_rpm_limit,
        )
    return _llm
