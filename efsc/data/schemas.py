from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class PromptExample:
    id: str
    family_id: str
    prompt: str
    language: str
    topic: str
    variant_type: str
    action_label: str
    harm_label: str
    legit_label: str
    uncertainty_label: str
    source: str = "unknown"
    source_split: str = "unknown"
    split: str = "train"
    preferred_over: List[str] = field(default_factory=list)
    dispreferred_over: List[str] = field(default_factory=list)
    sibling_ids: List[str] = field(default_factory=list)
    metadata: Optional[dict] = None
