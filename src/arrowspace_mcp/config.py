from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ServerConfig:
    allowed_paths: list[str] = field(default_factory=lambda: ["."])
