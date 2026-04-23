from typing import Dict

ACTION2ID: Dict[str, int] = {"ANSWER": 0, "SAFE_ANSWER": 1, "CLARIFY": 2, "REFUSE": 3}
ID2ACTION = {v: k for k, v in ACTION2ID.items()}

HARM2ID: Dict[str, int] = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}
LEGIT2ID: Dict[str, int] = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}
UNCERTAINTY2ID: Dict[str, int] = {"CLEAR": 0, "AMBIGUOUS": 1}

SOURCES = {
    "wildjailbreak",
    "falsereject",
    "xstest",
    "orbench_hard",
    "orbench_toxic",
    "strongreject",
    "custom_auth",
    "unknown",
}

VARIANT_TYPES = {
    "harmful_malicious",
    "defensive_benign",
    "educational_benign",
    "authorized_benign",
    "ambiguous_legitimate",
    "hard_benign",
    "toxic_harmful",
    "jailbreak",
    "paraphrase_benign",
    "translation_benign",
}

TOPICS = {"cyber", "bio", "fraud", "privacy", "mental_health", "chemistry", "general", "other"}
