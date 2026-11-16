"""Intent features: what the SMS asks the reader to do (not just money words)."""

from __future__ import annotations

import re
from dataclasses import dataclass
from urllib.parse import urlparse

import numpy as np

FEATURE_NAMES = [
    "asks_to_share_code",
    "warns_not_to_share",
    "has_link",
    "unofficial_link",
    "urgency_or_threat",
    "prize_or_too_good",
    "asks_to_send_money",
    "pure_notice",
]

# Domains that look like real Egyptian / global finance apps (synthetic whitelist).
OFFICIAL_HOST_MARKERS = (
    "vodafone.com",
    "vodafone.com.eg",
    "instapay.eg",
    "instapayegypt.com",
    "cibeg.com",
    "nbe.com.eg",
    "banquemisr.com",
    "orange.eg",
    "orange.com",
    "fawry.com",
    "etisalat.eg",
    "we.com.eg",
    "paypal.com",
    "chase.com",
    "bankofamerica.com",
    "wellsfargo.com",
    "venmo.com",
    "cash.app",
)

_LINK_RE = re.compile(r"https?://[^\s<>\"']+|www\.[^\s<>\"']+", re.I)

_SHARE_CODE = re.compile(
    r"("
    r"\b(send|share|forward|reply with|read aloud|tell me|give me)\b.{0,48}"
    r"\b(otp|pin|password|code|passcode|verification|secret)\b"
    r"|"
    r"\b(otp|pin|password|code|passcode)\b.{0,48}"
    r"\b(send|share|forward|reply|here|whatsapp)\b"
    r"|"
    r"(ابعت|ابعته|ابعتهالي|ابعتهولي|شارك|شاركوا|اكشف|اكتب|قول|اقرأ).{0,48}"
    r"(otp|كود|الرمز|رمز|الرقم السري|رقم السري|رقم سري|pin|باسورد|كلمة السر|صورة البطاقة)"
    r"|"
    r"(صورة البطاقة).{0,30}(رقم السري|الرقم السري|pin)"
    r"|"
    r"(eb3at|ebaat|eb3atly|share|e2ra|ekteb).{0,48}"
    r"(otp|pin|code|raqam el serry|el serry|password|soura el beta2a)"
    r")",
    re.I | re.DOTALL,
)

_WARN_NOT_SHARE = re.compile(
    r"("
    r"do not share|don't share|never share|do not give|don't give|"
    r"keep (this |your )?(code|otp|pin) private|"
    r"متشارك|متشاركش|لا تشارك|لا تشاركوا|متديش|لا تعط|"
    r"matsharksh|matshark|matdeesh|do not share / matsharksh"
    r")",
    re.I,
)

_URGENCY = re.compile(
    r"("
    r"\b(urgent|immediately|right now|within \d+ (min|mins|minutes|hours)|"
    r"today only|act now|will be (blocked|locked|suspended|closed)|"
    r"account will be|wallet will be)\b|"
    r"عاجل|فوراً|فورا|دلوقتي|هيتحظر|هيتقفل|هتتقفل|معلّقة|معلقة|خلال|"
    r"3agel|delwa2ty|hayet7azar|hayet2efel|hayet2afel|m3alla2a|fe \d+ de2i2a"
    r")",
    re.I,
)

_PRIZE = re.compile(
    r"("
    r"\b(won|winner|prize|congrats|congratulations|gift|claim|voucher|"
    r"jackpot|you have been selected)\b|"
    r"مبروك|كسبت|جائزة|جايزة|هدية|سحب|"
    r"mabrook|kesebt|gayza|hediya|prize"
    r")",
    re.I,
)

_SEND_MONEY = re.compile(
    r"("
    r"\b(send|transfer|pay|wire)\b.{0,30}\b(fee|fees|egp|money|usd|\$|pounds?)\b|"
    r"\b(activation fee|claim fee|unlock fee|processing|pay (a |the )?fee)\b|"
    r"(حوّل|حول|ابعت|ادفع|حوّلي|ابعتهالي).{0,30}"
    r"(ج|جنيه|رسوم|فلوس|تفعيل|فودافون كاش|انستاباي|أورانج كاش)|"
    r"(ابنك|ابن|علاج|محتاج).{0,40}(ابعت|حوّل|حول).{0,30}(ج|فودافون|انستاباي)|"
    r"(eb3at|edfa3|edfa3y|7awel)\b(?!t\b).{0,30}"
    r"(fees|floos|tafe3el|rasom|\d+\s*g\b|VF Cash|InstaPay|Orange Cash)|"
    r"\b7awel\b.{0,30}(\d+\s*g\b|fees|floos|VF Cash|InstaPay)|"
    r"(me7tag|mehtag|3elag).{0,40}(eb3at|7awel).{0,25}(g\b|VF Cash|InstaPay)"
    r")",
    re.I | re.DOTALL,
)

_COMPLETED_TRANSFER = re.compile(
    r"("
    r"\b(completed|successful|approved|received|you received)\b|"
    r"تم تحويل|تم استلام|تم قبول|بنجاح|"
    r"7awelt|estalamt|3amaliya tamam|successful"
    r")",
    re.I,
)

_OFFICIAL_PAY = re.compile(
    r"("
    r"official app|in the app|from the app|"
    r"التطبيق الرسمي|من التطبيق|"
    r"el app el rasmy|men el app el rasmy|official wallet app"
    r")",
    re.I,
)

_NOTICE = re.compile(
    r"("
    r"\b(received|you received|transfer .+ completed|approved|balance|"
    r"deposited|credited|receipt|paid successfully|no action needed|"
    r"notification only|as of today)\b|"
    r"تم استلام|تم تحويل|تم الموافقة|تم خصم|تم دفع|تم قبول|الرصيد|"
    r"رصيدك|رصيد|إيصال|الإخطار فقط|رسالة إشعار|"
    r"estalamt|estelam|balance|rasid|3amaliya tamam|notice bas|notification only"
    r")",
    re.I,
)

_MONEYISH = re.compile(
    r"("
    r"\b(otp|pin|egp|instapay|vodafone cash|orange cash|fawry|bank|wallet|"
    r"transfer|balance|salary|bill|card|fee|fees)\b|"
    r"انستاباي|فودافون|أورانج كاش|فوري|بنك|محفظ|تحويل|رصيد|راتب|فاتورة|بطاقة|ج |"
    r"otp|pin|instapay|vf cash|orange cash|fawry|rasid|7awal|beta2a"
    r")",
    re.I,
)


@dataclass
class IntentResult:
    vector: np.ndarray
    flags: dict[str, bool]
    reasons: list[str]


def _host_from_url(raw: str) -> str:
    url = raw if "://" in raw else "http://" + raw
    try:
        host = (urlparse(url).hostname or "").lower()
    except Exception:
        return ""
    if host.startswith("www."):
        host = host[4:]
    return host


def _is_official_host(host: str) -> bool:
    if not host:
        return False
    for marker in OFFICIAL_HOST_MARKERS:
        if host == marker or host.endswith("." + marker):
            return True
    return False


def extract_intent(text: str) -> IntentResult:
    t = text or ""
    links = _LINK_RE.findall(t)
    has_link = bool(links)
    unofficial = False
    for link in links:
        host = _host_from_url(link)
        if not _is_official_host(host):
            unofficial = True
            break

    warns = bool(_WARN_NOT_SHARE.search(t))
    asks_code = bool(_SHARE_CODE.search(t)) and not warns
    urgency = bool(_URGENCY.search(t))
    prize = bool(_PRIZE.search(t))
    official_pay = bool(_OFFICIAL_PAY.search(t))
    completed = bool(_COMPLETED_TRANSFER.search(t))
    send_money = bool(_SEND_MONEY.search(t)) and not official_pay and not completed
    notice = bool(_NOTICE.search(t)) or official_pay or completed
    # Pure notice: informative money SMS without scam asks.
    scamish = (
        asks_code
        or unofficial
        or prize
        or send_money
        or (urgency and (asks_code or send_money or unofficial))
    )
    pure_notice = (notice and not scamish) or (official_pay and not scamish and not asks_code)

    flags = {
        "asks_to_share_code": asks_code,
        "warns_not_to_share": warns,
        "has_link": has_link,
        "unofficial_link": unofficial,
        "urgency_or_threat": urgency,
        "prize_or_too_good": prize,
        "asks_to_send_money": send_money,
        "pure_notice": pure_notice,
    }
    vector = np.asarray([1.0 if flags[name] else 0.0 for name in FEATURE_NAMES], dtype=np.float32)
    reasons = intent_reasons(flags)
    return IntentResult(vector=vector, flags=flags, reasons=reasons)


def extract_intent_matrix(texts: list[str]) -> np.ndarray:
    if not texts:
        return np.zeros((0, len(FEATURE_NAMES)), dtype=np.float32)
    return np.vstack([extract_intent(t).vector for t in texts])


def intent_reasons(flags: dict[str, bool]) -> list[str]:
    """Plain-language reasons for the website."""
    reasons: list[str] = []
    if flags.get("asks_to_share_code"):
        reasons.append("Asks you to share a code, OTP, or PIN")
    if flags.get("unofficial_link"):
        reasons.append("Contains a link that is not an official domain")
    elif flags.get("has_link"):
        reasons.append("Contains a link")
    if flags.get("asks_to_send_money"):
        reasons.append("Asks you to send money or pay a fee")
    if flags.get("prize_or_too_good"):
        reasons.append("Promises a prize or something too good to be true")
    if flags.get("urgency_or_threat"):
        reasons.append("Uses urgency or a threat (block / act now)")
    if flags.get("warns_not_to_share"):
        reasons.append("Warns you not to share the code (common in real OTP notices)")
    if flags.get("pure_notice"):
        reasons.append("Only a notice — it does not ask you to do anything")
    if not reasons:
        reasons.append("No strong ask/intent phrases found")
    return reasons


def looks_like_money_message(text: str) -> bool:
    return bool(_MONEYISH.search(text or ""))
