"""Generation orchestration with retry and fallback."""

from __future__ import annotations

import asyncio
import time
from typing import List

import aiohttp

from .models import ImageModelTarget
from .providers import ImageGenerateRequest, ImageGenerateResult, create_adapter


IMAGE_RETRY_ATTEMPTS = 3


async def generate_image_with_fallback(
    targets: List[ImageModelTarget],
    req: ImageGenerateRequest,
    session: aiohttp.ClientSession,
) -> ImageGenerateResult:
    if not targets:
        return ImageGenerateResult(error="未配置生图模型")

    global_timeout = max(10, int(targets[0].timeout or 180))
    deadline = time.monotonic() + global_timeout
    last_error = "未配置生图模型"
    total_attempts = max(IMAGE_RETRY_ATTEMPTS, len(targets))

    for attempt in range(1, total_attempts + 1):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return ImageGenerateResult(error=f"生图全局超时（{global_timeout}秒），最后错误: {last_error}")

        target = targets[(attempt - 1) % len(targets)]
        label = target.label
        adapter = create_adapter(target, session)

        try:
            result = await asyncio.wait_for(adapter.generate(req), timeout=max(1, min(target.timeout, int(remaining))))
            if result.images and not result.error:
                result.used_model = label
                return result
            last_error = f"{label}: {result.error or '生成失败'}"
        except asyncio.TimeoutError:
            last_error = f"{label}: 请求超时"
        except Exception as exc:
            last_error = f"{label}: {exc}"

        if attempt < total_attempts:
            wait_seconds = attempt
            if deadline - time.monotonic() <= wait_seconds:
                return ImageGenerateResult(error=f"生图全局超时（{global_timeout}秒），最后错误: {last_error}")
            await asyncio.sleep(wait_seconds)

    return ImageGenerateResult(error=last_error)
