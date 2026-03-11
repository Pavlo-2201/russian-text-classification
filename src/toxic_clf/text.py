"""Минимальная нормализация текста.

Сознательно не удаляем пунктуацию, регистр, эмоджи и цифры: «!!!», «)))», капслок
и намеренные искажения могут нести сигнал токсичности. Решение о регистре принимается
параметром векторизатора и проверяется на validation.
"""

from __future__ import annotations

import re
import unicodedata

_WHITESPACE = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    """NFC, схлопывание пробельных символов (в т. ч. хвостового \\n) и обрезка краёв."""
    return _WHITESPACE.sub(" ", unicodedata.normalize("NFC", text)).strip()


def normalize_texts(texts) -> list[str]:
    return [normalize_text(t) for t in texts]
