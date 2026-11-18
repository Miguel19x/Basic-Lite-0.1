# Basic-Lite
# Copyright © 2026 Miguel Angel Aranguren Ramirez (Miguel19x)
# Todos los derechos reservados.
# Véanse LICENSE.txt y THIRD_PARTY_NOTICES.md.

import re
from typing import Optional, Tuple

_STRICT_WATT_MS_RE = re.compile(r"^watt:\s*(\d+)\s*\nms:\s*(\d+)\s*$", re.MULTILINE)
_STRICT_SCRATCHPAD_PW_RE = re.compile(
    r"^pw:\s*(\d+)\s*\npms:\s*(\d+)\s*\nwatt:\s*(\d+)\s*\nms:\s*(\d+)\s*$",
    re.MULTILINE,
)
_STRICT_SCRATCHPAD_SUB_RE = re.compile(
    r"^sub_w:\s*(\d+)\s*\nsub_ms:\s*(\d+)\s*\nwatt:\s*(\d+)\s*\nms:\s*(\d+)\s*$",
    re.MULTILINE,
)
_TOLERANT_SCRATCHPAD_RE = re.compile(
    r"(?:pw|sub_w):\s*(\d+).*?(?:pms|sub_ms):\s*(\d+).*?watt:\s*(\d+).*?ms:\s*(\d+)",
    re.DOTALL,
)
_TOLERANT_WATT_MS_RE = re.compile(r"watt:\s*(\d+).*?ms:\s*(\d+)", re.DOTALL)


def parse_watt_ms_prediction(
    generated_text: str, prompt_text: str
) -> Tuple[bool, Optional[int], Optional[int]]:
    """
    Extrae los enteros predichos para watt y ms de la región generada tras el prompt.
    Retorna: (es_formato_estricto, pred_watt, pred_ms)
    """
    continuation = (
        generated_text[len(prompt_text):]
        if generated_text.startswith(prompt_text)
        else generated_text
    )
    continuation_clean = continuation.strip()

    # 1. Extracción con traza scratchpad (pw o sub_w)
    m_scratch = _STRICT_SCRATCHPAD_PW_RE.match(continuation_clean) or _STRICT_SCRATCHPAD_SUB_RE.match(continuation_clean)
    if m_scratch:
        try:
            return True, int(m_scratch.group(3)), int(m_scratch.group(4))
        except ValueError:
            pass

    # 2. Extracción clásica estricta
    m_strict = _STRICT_WATT_MS_RE.match(continuation_clean)
    if m_strict:
        try:
            return True, int(m_strict.group(1)), int(m_strict.group(2))
        except ValueError:
            pass

    # 3. Extracción tolerante con scratchpad
    m_tol_scratch = _TOLERANT_SCRATCHPAD_RE.search(continuation)
    if m_tol_scratch:
        try:
            return False, int(m_tol_scratch.group(3)), int(m_tol_scratch.group(4))
        except ValueError:
            pass

    # 4. Extracción tolerante clásica
    m_tol = _TOLERANT_WATT_MS_RE.search(continuation)
    if m_tol:
        try:
            return False, int(m_tol.group(1)), int(m_tol.group(2))
        except ValueError:
            pass

    return False, None, None


def parse_scratchpad_prediction(
    generated_text: str, prompt_text: str
) -> Tuple[bool, Optional[int], Optional[int], Optional[int], Optional[int]]:
    """
    Extrae la traza completa de subtotales supervisados y resultados finales:
    Devuelve (strict_ok, pred_sub_w, pred_sub_ms, pred_watt, pred_ms).
    """
    continuation = (
        generated_text[len(prompt_text):]
        if generated_text.startswith(prompt_text)
        else generated_text
    )
    continuation_clean = continuation.strip()

    # 1. Coincidencia estricta con scratchpad (pw o sub_w)
    m_scratch = _STRICT_SCRATCHPAD_PW_RE.match(continuation_clean) or _STRICT_SCRATCHPAD_SUB_RE.match(continuation_clean)
    if m_scratch:
        try:
            return (
                True,
                int(m_scratch.group(1)),
                int(m_scratch.group(2)),
                int(m_scratch.group(3)),
                int(m_scratch.group(4)),
            )
        except ValueError:
            pass

    # 2. Coincidencia tolerante con scratchpad
    m_tol_scratch = _TOLERANT_SCRATCHPAD_RE.search(continuation)
    if m_tol_scratch:
        try:
            return (
                False,
                int(m_tol_scratch.group(1)),
                int(m_tol_scratch.group(2)),
                int(m_tol_scratch.group(3)),
                int(m_tol_scratch.group(4)),
            )
        except ValueError:
            pass

    # 3. Formato clásico sin subtotales
    m_strict = _STRICT_WATT_MS_RE.match(continuation_clean)
    if m_strict:
        try:
            return True, None, None, int(m_strict.group(1)), int(m_strict.group(2))
        except ValueError:
            pass

    m_tol = _TOLERANT_WATT_MS_RE.search(continuation)
    if m_tol:
        try:
            return False, None, None, int(m_tol.group(1)), int(m_tol.group(2))
        except ValueError:
            pass

    return False, None, None, None, None
