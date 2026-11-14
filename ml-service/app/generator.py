"""Generate a separate stand-alone draft for a confirmed target category via vLLM."""
import json
import logging
import os
import re
import httpx

LOG = logging.getLogger(__name__)
VLLM_URL = os.getenv("VLLM_URL", "http://127.0.0.1:8000").rstrip("/")
VLLM_MODEL = os.getenv("VLLM_MODEL", "avito-qwen")
LABELS = {
    102: "Штукатурные работы",
    103: "Плиточные работы",
    104: "Сантехника",
    105: "Электрика",
    106: "Поклейка обоев и малярные работы",
}
GUIDE = {
    102: "Оштукатуривание и выравнивание штукатурными составами, включая откосы и маяки. Не путай со шпаклёвкой под покраску.",
    103: "Плитка, керамогранит, мозаика, фартук, облицовка, затирка плиточных швов.",
    104: "Монтаж и ремонт сантехники, труб, отопления, подключение техники к воде и сливу.",
    105: "Проводка, розетки, выключатели, электрощиты, светильники, электрический тёплый пол.",
    106: "Обои, покраска, шпаклёвка под покраску, подготовка к малярной отделке.",
}

FALLBACK_TITLES = {
    102: "Штукатурные работы",
    103: "Плиточные работы",
    104: "Сантехнические работы",
    105: "Электромонтажные работы",
    106: "Поклейка обоев и малярные работы",
}

# Only accept fallback quotes that explicitly mention a related service.
# This is a conservative lexical check, not a complete semantic verifier.
SERVICE_HINTS = {
    102: (r"штукатур", r"оштукатур", r"маяк"),
    103: (r"плитк", r"плиточ", r"керамогранит", r"мозаик", r"облицов"),
    104: (r"сантех", r"раковин", r"унитаз", r"водоснабж", r"канализац", r"смесител", r"труб"),
    105: (r"электр", r"проводк", r"розет", r"выключател", r"электрощит", r"светильник"),
    106: (r"обо[ийв]", r"маляр", r"краси", r"покрас", r"шпакл", r"шпатл"),
}


def fallback_from_evidence(description: str, evidence: object, mc_id: int):
    """Use only the source quote, not potentially invented generated facts."""
    if not isinstance(evidence, str) or not evidence.strip():
        return None
    quote = evidence.strip()
    if quote not in description or len(quote) > 500:
        return None
    if not any(re.search(p, quote, re.I) for p in SERVICE_HINTS[mc_id]):
        return None
    # Obvious denials or other contractors are not valid service evidence.
    if re.search(r"\b(?:не\s+(?:делаем|выполняем|занимаемся|оказываем|клеим)|друг(?:ой|ие)\s+мастер|заказчик\s+сам|не\s+предлагаем)\b", quote, re.I):
        return None
    return FALLBACK_TITLES[mc_id], quote[0].upper() + quote[1:]


SCHEMA = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "text": {"type": "string"},
        "evidence": {"type": "string"},
    },
    "required": ["title", "text", "evidence"],
    "additionalProperties": False,
}


def validate_draft(title: object, text: object, evidence: object, description: str) -> bool:
    """Conservative checks; cannot establish semantic faithfulness completely."""
    if not all(isinstance(x, str) and x.strip() for x in (title, text, evidence)):
        return False
    if len(title) > 50 or len(text) > 1500 or len(evidence) > 800:
        return False
    # The quote must be exact; failed drafts remain valid as null fields in the public API.
    if evidence not in description:
        return False
    if any(phrase in text.lower() for phrase in ("исходном объявлении", "из объявления", "отдельно заказать")):
        return False
    # Reject unsupported quantified promises; not a complete hallucination detector.
    forbidden = (r"\bгаранти\w*", r"\b\d+\s*лет\s*опыта\b", r"\bскидк\w*", r"\bбесплатн\w*")
    for pattern in forbidden:
        if re.search(pattern, title + " " + text, re.I) and not re.search(pattern, description, re.I):
            return False
    return True


async def generate_draft(client: httpx.AsyncClient, description: str, mc_id: int) -> tuple[str | None, str | None]:
    messages = [
        {"role": "system", "content": (
            "Ты составляешь один самостоятельный черновик объявления Авито строго в указанной микрокатегории. "
            "Используй ТОЛЬКО явно предложенные исполнителем работы и факты из исходного текста. "
            "Игнорируй отрицания, чужие работы, предположения и сравнения. "
            "Не придумывай цены, гарантии, опыт, сроки, скидки, отдельный заказ или иные условия. "
            "Не ссылайся на исходное объявление. Заголовок — до 50 символов, текст — до 1500. "
            "evidence — одна дословная непрерывная цитата из исходного текста, доказывающая целевую услугу. "
            "Если такой цитаты нет, верни пустые строки title, text, evidence. Верни только JSON."
        )},
        {"role": "user", "content": (
            f"Целевая категория {mc_id}: {LABELS[mc_id]}.\n"
            f"Что относится к ней: {GUIDE[mc_id]}\n"
            f"Объявление:\n{description}"
        )},
    ]
    try:
        response = await client.post(
            f"{VLLM_URL}/v1/chat/completions",
            json={
                "model": VLLM_MODEL,
                "messages": messages,
                "temperature": 0,
                "max_tokens": 550,
                "structured_outputs": {"json": SCHEMA},
            },
        )
        response.raise_for_status()
        completion = response.json()["choices"][0]
        if completion.get("finish_reason") == "length":
            LOG.warning("Generation truncated for %s", mc_id)
            return None, None
        output = json.loads(completion["message"]["content"])
        title, text, evidence = output.get("title"), output.get("text"), output.get("evidence")
        if validate_draft(title, text, evidence, description):
            return title.strip(), text.strip()
        # Frequently the model produces an empty title despite a genuine quote.
        # Never reuse its unsupported prose on this fallback path.
        fallback = fallback_from_evidence(description, evidence, mc_id) if not (isinstance(title, str) and title.strip()) else None
        if fallback is not None:
            LOG.info("Source-grounded fallback used for mcId=%s", mc_id)
            return fallback
        LOG.warning("Generated draft rejected for mcId=%s (title=%r, evidence_exact=%s)",
                    mc_id, title, isinstance(evidence, str) and evidence in description)
        return None, None
    except (httpx.HTTPError, ValueError, KeyError, IndexError, TypeError) as exc:
        LOG.warning("Generation failed for mcId=%s: %s", mc_id, exc)
        return None, None
