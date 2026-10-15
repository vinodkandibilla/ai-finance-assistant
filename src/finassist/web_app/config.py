"""Frontend configuration, without loading backend settings or secrets."""

import math
import os
from dataclasses import dataclass
from urllib.parse import urlsplit


@dataclass(frozen=True)
class WebSettings:
    api_base_url: str = "http://127.0.0.1:8000"
    connect_timeout: float = 5.0
    read_timeout: float = 120.0
    write_timeout: float = 10.0
    pool_timeout: float = 5.0
    health_timeout: float = 5.0

    def __post_init__(self) -> None:
        url = self.api_base_url.rstrip("/")
        try:
            parsed = urlsplit(url)
            valid = parsed.scheme in {"http", "https"} and parsed.hostname and parsed.port != 0
            if not valid or parsed.username or parsed.password or parsed.query or parsed.fragment:
                raise ValueError
        except ValueError:
            raise ValueError(
                "API base URL must be an HTTP(S) URL without credentials or query."
            ) from None
        object.__setattr__(self, "api_base_url", url)
        for value in (
            self.connect_timeout,
            self.read_timeout,
            self.write_timeout,
            self.pool_timeout,
            self.health_timeout,
        ):
            if not math.isfinite(value) or value <= 0:
                raise ValueError("API timeouts must be finite positive seconds.")

    @classmethod
    def from_environment(cls) -> "WebSettings":
        try:
            return cls(
                api_base_url=os.getenv("FINPILOT_API_BASE_URL", "http://127.0.0.1:8000"),
                **{
                    name: float(os.getenv(f"FINPILOT_{name.upper()}", str(default)))
                    for name, default in {
                        "connect_timeout": 5,
                        "read_timeout": 120,
                        "write_timeout": 10,
                        "pool_timeout": 5,
                        "health_timeout": 5,
                    }.items()
                },
            )
        except ValueError:
            raise ValueError(
                "Check FINPILOT_API_BASE_URL and FINPILOT_*_TIMEOUT settings."
            ) from None
