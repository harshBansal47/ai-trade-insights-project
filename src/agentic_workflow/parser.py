# src/agents/parser.py

"""
Signal Parser
-------------
Responsible for parsing, validating, and sanitising the raw LLM output
from the signal agent into a guaranteed-clean SignalOutput.

Handles all failure modes:
  - Raw JSON string with markdown fences
  - Partial / incomplete JSON from truncated LLM responses
  - Type coercion errors (string numbers, wrong enums)
  - Missing optional fields
  - Hallucinated price levels (nulled out when underivable)
  - Confidence clamping and risk_level auto-derivation
"""

from __future__ import annotations

import json
import re
from typing import Any, Optional

from pydantic import ValidationError


from agentic_workflow.schemas import RiskLevel, SignalOutput, SignalType
from src.core.logger import setup_logger

logger = setup_logger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Signal → minimum sensible confidence floor
_SIGNAL_CONFIDENCE_FLOOR: dict[str, float] = {
    SignalType.STRONG_BUY:  0.70,
    SignalType.BUY:         0.45,
    SignalType.NEUTRAL:     0.00,
    SignalType.SELL:        0.45,
    SignalType.STRONG_SELL: 0.70,
    SignalType.NO_TRADE:    0.00,
}

# Signals that must NEVER have a trade_setup attached
_NO_SETUP_SIGNALS = {SignalType.NEUTRAL, SignalType.NO_TRADE}

# Required confluence factor labels — at least one of each must be present
_REQUIRED_CONFLUENCE_FACTORS = {"Trend", "RSI", "MACD", "BB", "SR"}

# Valid alignment values for confluence factors
_VALID_ALIGNMENTS = {"bullish", "bearish", "neutral", "warning"}

# Valid bias values for timeframe signals
_VALID_BIASES = {"bullish", "bearish", "neutral"}

# Valid strength values
_VALID_STRENGTHS = {"strong", "moderate", "weak"}


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def parse_signal_output(raw: Any, report: dict) -> Optional[SignalOutput]:
    """
    Parse and validate the LLM output into a clean SignalOutput.

    Accepts:
      - A SignalOutput instance (already parsed by LangChain's PydanticOutputParser)
      - A raw string (JSON with or without markdown fences)
      - A dict (pre-parsed JSON)

    Parameters
    ----------
    raw    : LLM output in any of the above forms.
    report : Original report dict — used to cross-validate price levels
             and fill missing timeframe entries.

    Returns
    -------
    SignalOutput if parsing succeeds, None otherwise.
    """
    logger.debug("[PARSER] Starting parse")

    try:
        data = _extract_dict(raw)
        if data is None:
            logger.error("[PARSER] Could not extract dict from raw output")
            return None

        data = _sanitise(data, report)
        signal = _build_model(data)

        logger.info(
            f"[PARSER] Parsed signal={signal.signal} "
            f"confidence={signal.confidence} "
            f"fake_warning={signal.fake_signal_warning}"
        )
        return signal

    except Exception as exc:
        logger.error(f"[PARSER] Unexpected failure: {exc}", exc_info=True)
        return None


# ---------------------------------------------------------------------------
# Step 1 — Extract a plain dict from whatever the LLM returned
# ---------------------------------------------------------------------------

def _extract_dict(raw: Any) -> Optional[dict]:
    """Convert any LLM output form into a plain Python dict."""

    # Already a Pydantic model — convert directly
    if isinstance(raw, SignalOutput):
        return raw.model_dump()

    # Already a dict
    if isinstance(raw, dict):
        return raw

    # String — strip markdown fences and parse JSON
    if isinstance(raw, str):
        return _parse_json_string(raw)

    # LangChain AIMessage or similar — extract .content
    if hasattr(raw, "content"):
        return _parse_json_string(raw.content)

    logger.warning(f"[PARSER] Unrecognised raw type: {type(raw)}")
    return None


def _parse_json_string(text: str) -> Optional[dict]:
    """
    Extract and parse JSON from a string, handling:
      - Bare JSON
      - ```json ... ``` fences
      - ``` ... ``` fences
      - Leading/trailing whitespace and commentary
    """
    text = text.strip()

    # Strip markdown code fences
    fence_match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text, re.IGNORECASE)
    if fence_match:
        text = fence_match.group(1).strip()

    # Find the outermost JSON object
    obj_match = re.search(r"\{[\s\S]*\}", text)
    if not obj_match:
        logger.error("[PARSER] No JSON object found in string")
        return None

    try:
        return json.loads(obj_match.group())
    except json.JSONDecodeError as exc:
        logger.error(f"[PARSER] JSON decode error: {exc}")

        # Last resort — attempt to salvage truncated JSON by closing braces
        salvaged = _salvage_truncated_json(obj_match.group())
        if salvaged:
            logger.warning("[PARSER] Salvaged truncated JSON")
            return salvaged

        return None


def _salvage_truncated_json(text: str) -> Optional[dict]:
    """
    Try to recover truncated JSON by closing unclosed braces/brackets.
    Only used as a last resort for truncated LLM responses.
    """
    open_braces   = text.count("{") - text.count("}")
    open_brackets = text.count("[") - text.count("]")

    # Strip trailing comma if present before closing
    text = re.sub(r",\s*$", "", text.rstrip())

    text += "]" * max(open_brackets, 0)
    text += "}" * max(open_braces, 0)

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


# ---------------------------------------------------------------------------
# Step 2 — Sanitise and normalise the dict before Pydantic validation
# ---------------------------------------------------------------------------

def _sanitise(data: dict, report: dict) -> dict:
    """Run all sanitisation passes in order."""
    data = _coerce_types(data)
    data = _clamp_confidence(data)
    data = _derive_risk_level(data)
    data = _enforce_no_setup_on_neutral(data)
    data = _validate_trade_setup_levels(data, report)
    data = _ensure_timeframe_coverage(data, report)
    data = _ensure_confluence_coverage(data)
    data = _sanitise_confluence_alignments(data)
    data = _sanitise_timeframe_biases(data)
    data = _enforce_strong_signal_fake_rule(data)
    data = _sanitise_caution(data)
    return data


# ── Type coercion ──────────────────────────────────────────────────────────

def _coerce_types(data: dict) -> dict:
    """Coerce common type mismatches from LLM output."""

    # confidence: sometimes returned as string "0.72" or percentage "72"
    if "confidence" in data:
        try:
            val = float(data["confidence"])
            # If given as 0–100 instead of 0–1
            data["confidence"] = val / 100.0 if val > 1.0 else val
        except (TypeError, ValueError):
            data["confidence"] = 0.0

    # signal: normalise case
    if "signal" in data and isinstance(data["signal"], str):
        data["signal"] = data["signal"].upper().strip()

    # fake_signal_warning: sometimes returned as string
    if "fake_signal_warning" in data:
        val = data["fake_signal_warning"]
        if isinstance(val, str):
            data["fake_signal_warning"] = val.lower() in ("true", "1", "yes")

    # risk_level: normalise case
    if "risk_level" in data and isinstance(data["risk_level"], str):
        data["risk_level"] = data["risk_level"].upper().strip()

    # trade_setup floats: coerce string numbers
    if isinstance(data.get("trade_setup"), dict):
        ts = data["trade_setup"]
        for field in ("stop_loss", "take_profit_1", "take_profit_2",
                      "take_profit_3", "risk_reward"):
            if field in ts and ts[field] is not None:
                try:
                    ts[field] = float(ts[field])
                except (TypeError, ValueError):
                    ts[field] = None

        if isinstance(ts.get("entry_zone"), dict):
            for bound in ("low", "high"):
                if bound in ts["entry_zone"] and ts["entry_zone"][bound] is not None:
                    try:
                        ts["entry_zone"][bound] = float(ts["entry_zone"][bound])
                    except (TypeError, ValueError):
                        ts["entry_zone"][bound] = None

    return data


# ── Confidence clamping ────────────────────────────────────────────────────

def _clamp_confidence(data: dict) -> dict:
    """
    Clamp confidence to [0.0, 1.0] and enforce minimum floors per signal type.
    The LLM is instructed to use abs(confidence_score), but we enforce it here.
    """
    raw_conf = float(data.get("confidence", 0.0))
    clamped  = max(0.0, min(1.0, raw_conf))

    signal   = data.get("signal", SignalType.NEUTRAL)
    floor    = _SIGNAL_CONFIDENCE_FLOOR.get(signal, 0.0)

    data["confidence"] = round(max(clamped, floor), 3)
    return data


# ── Risk level derivation ──────────────────────────────────────────────────

def _derive_risk_level(data: dict) -> dict:
    """
    Auto-derive risk_level from signal + fake_signal_warning if missing
    or invalid.
    """
    valid_levels = {r.value for r in RiskLevel}
    current = str(data.get("risk_level", "")).upper()

    if current in valid_levels:
        return data  # LLM got it right

    # Derive from signal and fake warning
    signal       = data.get("signal", SignalType.NEUTRAL)
    fake_warning = bool(data.get("fake_signal_warning", False))

    if fake_warning:
        data["risk_level"] = RiskLevel.HIGH
    elif signal in (SignalType.STRONG_BUY, SignalType.STRONG_SELL):
        data["risk_level"] = RiskLevel.LOW
    elif signal in (SignalType.BUY, SignalType.SELL):
        data["risk_level"] = RiskLevel.MODERATE
    else:
        data["risk_level"] = RiskLevel.HIGH

    logger.debug(f"[PARSER] risk_level derived as {data['risk_level']}")
    return data


# ── Trade setup on neutral/no-trade ───────────────────────────────────────

def _enforce_no_setup_on_neutral(data: dict) -> dict:
    """Hard rule 4: NEUTRAL and NO_TRADE must never have a trade_setup."""
    signal = data.get("signal", "")
    if signal in _NO_SETUP_SIGNALS or signal in {s.value for s in _NO_SETUP_SIGNALS}:
        if data.get("trade_setup") is not None:
            logger.warning(
                f"[PARSER] Removing trade_setup for signal={signal} (hard rule 4)"
            )
            data["trade_setup"] = None
    return data


# ── Price level validation ─────────────────────────────────────────────────

def _validate_trade_setup_levels(data: dict, report: dict) -> dict:
    """
    Cross-validate trade_setup price levels against SR zones in the report.
    Null out levels that appear completely hallucinated (outside any known
    SR zone by more than 15%).

    Only a sanity check — not a strict clamp — because different TFs have
    different price scales and the LLM may be synthesising across them.
    """
    ts = data.get("trade_setup")
    if not ts or not isinstance(ts, dict):
        return data

    all_levels = _collect_all_sr_levels(report)
    if not all_levels:
        return data  # no reference levels available, skip check

    min_level = min(all_levels) * 0.85
    max_level = max(all_levels) * 1.15

    def _check(val: Optional[float], field: str) -> Optional[float]:
        if val is None:
            return None
        if not (min_level <= val <= max_level):
            logger.warning(
                f"[PARSER] Nulling hallucinated price level "
                f"field={field} value={val} "
                f"range=[{min_level:.2f}, {max_level:.2f}]"
            )
            return None
        return val

    ez = ts.get("entry_zone", {})
    if isinstance(ez, dict):
        ez["low"]  = _check(ez.get("low"),  "entry_zone.low")
        ez["high"] = _check(ez.get("high"), "entry_zone.high")

    ts["stop_loss"]     = _check(ts.get("stop_loss"),     "stop_loss")
    ts["take_profit_1"] = _check(ts.get("take_profit_1"), "take_profit_1")
    ts["take_profit_2"] = _check(ts.get("take_profit_2"), "take_profit_2")
    ts["take_profit_3"] = _check(ts.get("take_profit_3"), "take_profit_3")

    return data


def _collect_all_sr_levels(report: dict) -> list[float]:
    """Flatten all SR levels from every timeframe in the report."""
    levels: list[float] = []
    for tf_data in report.get("timeframes", {}).values():
        sr = tf_data.get("sr_zones", {})
        for zone_list in (sr.get("resistance", []), sr.get("support", [])):
            for zone in zone_list:
                if isinstance(zone.get("level"), (int, float)):
                    levels.append(float(zone["level"]))
    return levels


# ── Timeframe coverage ─────────────────────────────────────────────────────

def _ensure_timeframe_coverage(data: dict, report: dict) -> dict:
    """
    Hard rule 6: every timeframe in the report must appear in
    timeframe_signals. Fill any missing ones with a neutral placeholder.
    """
    existing_tfs = {
        entry["timeframe"]
        for entry in data.get("timeframe_signals", [])
        if isinstance(entry, dict) and "timeframe" in entry
    }

    report_tfs = set(report.get("timeframes", {}).keys())
    missing    = report_tfs - existing_tfs

    if missing:
        logger.warning(f"[PARSER] Missing timeframe entries: {missing} — filling neutral")
        for tf in sorted(missing):
            data.setdefault("timeframe_signals", []).append({
                "timeframe": tf,
                "bias":      "neutral",
                "strength":  "weak",
                "key_note":  "No signal data available for this timeframe",
            })

    return data


# ── Confluence factor coverage ─────────────────────────────────────────────

def _ensure_confluence_coverage(data: dict) -> dict:
    """
    Hard rule 7: at least one confluence factor per required label.
    Append a neutral placeholder for any missing factor.
    """
    factors   = data.get("confluence_factors", [])
    present   = {
        f["factor"] for f in factors
        if isinstance(f, dict) and "factor" in f
    }
    missing   = _REQUIRED_CONFLUENCE_FACTORS - present

    if missing:
        logger.warning(f"[PARSER] Missing confluence factors: {missing} — filling neutral")
        for label in sorted(missing):
            factors.append({
                "factor":    label,
                "alignment": "neutral",
                "detail":    "Insufficient data to assess this factor",
            })

    data["confluence_factors"] = factors
    return data


# ── Confluence alignment sanitisation ─────────────────────────────────────

def _sanitise_confluence_alignments(data: dict) -> dict:
    """Clamp any invalid alignment values to 'neutral'."""
    for factor in data.get("confluence_factors", []):
        if not isinstance(factor, dict):
            continue
        if factor.get("alignment") not in _VALID_ALIGNMENTS:
            logger.debug(
                f"[PARSER] Invalid confluence alignment "
                f"'{factor.get('alignment')}' → 'neutral'"
            )
            factor["alignment"] = "neutral"
    return data


# ── Timeframe bias sanitisation ────────────────────────────────────────────

def _sanitise_timeframe_biases(data: dict) -> dict:
    """Clamp invalid bias/strength values in timeframe_signals."""
    for entry in data.get("timeframe_signals", []):
        if not isinstance(entry, dict):
            continue
        if entry.get("bias") not in _VALID_BIASES:
            entry["bias"] = "neutral"
        if entry.get("strength") not in _VALID_STRENGTHS:
            entry["strength"] = "weak"
    return data


# ── STRONG signal + fake_signal_warning veto ──────────────────────────────

def _enforce_strong_signal_fake_rule(data: dict) -> dict:
    """
    Hard rule 3: STRONG_BUY and STRONG_SELL must never coexist with
    fake_signal_warning = true. Downgrade to BUY / SELL if violated.
    """
    signal       = data.get("signal", "")
    fake_warning = bool(data.get("fake_signal_warning", False))

    if fake_warning and signal == SignalType.STRONG_BUY:
        logger.warning("[PARSER] Downgrading STRONG_BUY → BUY (fake_signal_warning=true)")
        data["signal"]     = SignalType.BUY
        data["confidence"] = min(data.get("confidence", 0.6), 0.69)

    elif fake_warning and signal == SignalType.STRONG_SELL:
        logger.warning("[PARSER] Downgrading STRONG_SELL → SELL (fake_signal_warning=true)")
        data["signal"]     = SignalType.SELL
        data["confidence"] = min(data.get("confidence", 0.6), 0.69)

    return data


# ── Caution field sanitisation ─────────────────────────────────────────────

def _sanitise_caution(data: dict) -> dict:
    """
    Hard rule 9: caution must be a single sentence string or null.
    Collapse lists into a joined string; truncate overly long values.
    """
    caution = data.get("caution")

    if caution is None or caution == "":
        data["caution"] = None
        return data

    # LLM sometimes returns a list
    if isinstance(caution, list):
        caution = ". ".join(str(c).strip(". ") for c in caution if c) + "."
        logger.debug("[PARSER] Joined caution list into single string")

    caution = str(caution).strip()

    # Truncate if excessively long (> 300 chars)
    if len(caution) > 300:
        caution = caution[:297].rsplit(" ", 1)[0] + "..."
        logger.debug("[PARSER] Truncated caution to 300 chars")

    data["caution"] = caution if caution else None
    return data


# ---------------------------------------------------------------------------
# Step 3 — Build and validate the Pydantic model
# ---------------------------------------------------------------------------

def _build_model(data: dict) -> SignalOutput:
    """
    Attempt to build a SignalOutput from sanitised data.
    On ValidationError, attempt field-by-field recovery before raising.
    """
    try:
        return SignalOutput(**data)

    except ValidationError as exc:
        logger.warning(f"[PARSER] Pydantic validation failed — attempting recovery: {exc}")
        data = _recover_from_validation_error(data, exc)

        try:
            return SignalOutput(**data)
        except ValidationError as final_exc:
            logger.error(f"[PARSER] Recovery failed: {final_exc}")
            raise


def _recover_from_validation_error(data: dict, exc: ValidationError) -> dict:
    """
    Field-by-field recovery for common Pydantic validation errors.
    Attempts to fix or null out each failing field.
    """
    for error in exc.errors():
        field_path = " → ".join(str(loc) for loc in error["loc"])
        error_type = error["type"]

        logger.debug(f"[PARSER] Recovery attempt: field={field_path} type={error_type}")

        # Enum errors — try to coerce or fall back to safe default
        if error_type == "enum":
            _fix_enum_field(data, error["loc"])

        # Missing required field — insert safe default
        elif error_type == "missing":
            _fix_missing_field(data, error["loc"])

        # Type error on a numeric field — null it
        elif error_type in ("float_type", "int_type", "decimal_type"):
            _null_numeric_field(data, error["loc"])

    return data


def _fix_enum_field(data: dict, loc: tuple) -> None:
    """Attempt to coerce an invalid enum value, else set a safe default."""
    field = loc[0]

    if field == "signal":
        data["signal"] = SignalType.NO_TRADE
    elif field == "risk_level":
        data["risk_level"] = RiskLevel.HIGH
    elif len(loc) >= 2:
        # Nested enum e.g. timeframe_signals[0].bias
        parent, index, child = loc[0], loc[1], loc[2] if len(loc) > 2 else None
        try:
            if child and isinstance(data.get(parent), list):
                entry = data[parent][int(index)]
                if child == "bias":
                    entry["bias"] = "neutral"
                elif child == "strength":
                    entry["strength"] = "weak"
                elif child == "alignment":
                    entry["alignment"] = "neutral"
        except (IndexError, KeyError, TypeError):
            pass


def _fix_missing_field(data: dict, loc: tuple) -> None:
    """Insert a safe default for a missing required field."""
    field = loc[0]

    defaults = {
        "symbol":              "UNKNOWN",
        "mode":                "UNKNOWN",
        "signal":              SignalType.NO_TRADE,
        "confidence":          0.0,
        "risk_level":          RiskLevel.HIGH,
        "headline":            "Signal data incomplete",
        "summary":             "Insufficient data to generate a complete signal.",
        "fake_signal_warning": False,
        "dominant_trend":      "SIDEWAYS",
        "bb_context":          "BB context unavailable",
        "timeframe_signals":   [],
        "confluence_factors":  [],
    }

    if field in defaults and field not in data:
        data[field] = defaults[field]
        logger.debug(f"[PARSER] Inserted default for missing field: {field}")


def _null_numeric_field(data: dict, loc: tuple) -> None:
    """Null out a numeric field that failed type validation."""
    try:
        if len(loc) == 1:
            data[loc[0]] = None
        elif len(loc) == 2:
            parent, child = loc
            if isinstance(data.get(parent), dict):
                data[parent][child] = None
        elif len(loc) == 3:
            parent, index, child = loc
            if isinstance(data.get(parent), list):
                data[parent][int(index)][child] = None
    except (KeyError, IndexError, TypeError):
        pass