"""Self-play and cross-play data generation for offline knowledge distillation.

Generates synthetic Moshi conversations capturing teacher targets (logits,
tokens) at each step for offline distillation.

- **Self-play:** One model talks to itself (output looped back as input).
- **Cross-play:** Two independent models converse in full-duplex mode,
  producing higher-quality training data.
"""

from .generator import (  # noqa: F401
    # Self-play
    ConversationRecord,
    generate_conversation,
    save_conversation,
    generate_batch,
    # Cross-play
    CrossPlayRecord,
    generate_cross_play_conversation,
    save_cross_play_conversation,
    generate_cross_play_batch,
)
from .quality_filter import filter_conversation, QualityReport  # noqa: F401
