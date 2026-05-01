from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from enum import Enum
import json
import logging
from logging.handlers import RotatingFileHandler
import os
from pathlib import Path
from typing import Any
from uuid import UUID


def _env_flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _resolve_log_file_path(env_var: str) -> Path | None:
    raw = os.getenv(env_var)
    if raw is None or not raw.strip():
        return None
    return Path(raw).expanduser().resolve()


def _default_failure_log_path(main_log_path: Path | None) -> Path | None:
    raw = os.getenv("LOG_FAILURE_FILE_PATH")
    if raw is not None:
        return Path(raw).expanduser().resolve() if raw.strip() else None
    if main_log_path is None:
        return None
    return main_log_path.with_name(
        f"{main_log_path.stem}_failures{main_log_path.suffix}"
    )


def _rotating_file_handler(
    path: Path,
    formatter: logging.Formatter,
) -> RotatingFileHandler:
    path.parent.mkdir(parents=True, exist_ok=True)
    handler = RotatingFileHandler(
        path,
        maxBytes=_env_int("LOG_MAX_BYTES", 10 * 1024 * 1024),
        backupCount=_env_int("LOG_BACKUP_COUNT", 5),
        encoding="utf-8",
    )
    handler.setFormatter(formatter)
    return handler


def _serialize_log_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (datetime, UUID, Enum)):
        return str(value)
    if isinstance(value, Mapping):
        return {
            str(key): _serialize_log_value(item)
            for key, item in value.items()
            if item is not None
        }
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_serialize_log_value(item) for item in value if item is not None]
    return str(value)


class JsonLogFormatter(logging.Formatter):
    def __init__(self, *, service_name: str) -> None:
        super().__init__()
        self.service_name = service_name

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "service": self.service_name,
            "message": record.getMessage(),
        }

        event = getattr(record, "event", None)
        if event:
            payload["event"] = str(event)

        structured_fields = getattr(record, "structured_fields", None)
        if isinstance(structured_fields, Mapping):
            for key, value in structured_fields.items():
                serialized = _serialize_log_value(value)
                if serialized is not None:
                    payload[str(key)] = serialized

        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)

        return json.dumps(payload, ensure_ascii=True, separators=(",", ":"))


class KeyValueLogFormatter(logging.Formatter):
    def __init__(self, *, service_name: str) -> None:
        super().__init__()
        self.service_name = service_name

    def format(self, record: logging.LogRecord) -> str:
        parts = [
            f"timestamp={datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat()}",
            f"level={record.levelname}",
            f"service={self.service_name}",
            f"logger={record.name}",
        ]

        event = getattr(record, "event", None)
        if event:
            parts.append(f"event={event}")

        parts.append(f'message="{record.getMessage()}"')

        structured_fields = getattr(record, "structured_fields", None)
        if isinstance(structured_fields, Mapping):
            for key, value in structured_fields.items():
                serialized = _serialize_log_value(value)
                if serialized is None:
                    continue
                serialized_text = json.dumps(serialized, ensure_ascii=True)
                parts.append(f"{key}={serialized_text}")

        if record.exc_info:
            parts.append(f'exception="{self.formatException(record.exc_info)}"')

        return " ".join(parts)


def configure_runtime_logging(*, service_name: str, log_level: str) -> None:
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.setLevel(log_level)
    access_log_enabled = _env_flag("UVICORN_ACCESS_LOG", False)

    log_format = os.getenv("LOG_FORMAT", "json").strip().lower()
    formatter: logging.Formatter
    if log_format == "text":
        formatter = KeyValueLogFormatter(service_name=service_name)
    else:
        formatter = JsonLogFormatter(service_name=service_name)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    root_logger.addHandler(stream_handler)

    resolved_log_path = _resolve_log_file_path("LOG_FILE_PATH")
    if resolved_log_path is not None:
        root_logger.addHandler(_rotating_file_handler(resolved_log_path, formatter))

    failure_log_path = _default_failure_log_path(resolved_log_path)
    if failure_log_path is not None:
        failure_handler = _rotating_file_handler(failure_log_path, formatter)
        failure_handler.setLevel(logging.WARNING)
        root_logger.addHandler(failure_handler)

    logging.captureWarnings(True)
    for logger_name in ("uvicorn", "uvicorn.error"):
        uvicorn_logger = logging.getLogger(logger_name)
        uvicorn_logger.handlers.clear()
        uvicorn_logger.propagate = True
        uvicorn_logger.setLevel(log_level)

    uvicorn_access_logger = logging.getLogger("uvicorn.access")
    uvicorn_access_logger.handlers.clear()
    uvicorn_access_logger.propagate = access_log_enabled
    uvicorn_access_logger.disabled = not access_log_enabled
    uvicorn_access_logger.setLevel(log_level)


def log_event(
    logger: logging.Logger,
    level: int,
    event: str,
    *,
    message: str | None = None,
    exc_info: bool | BaseException | tuple[type[BaseException], BaseException, Any] | None = None,
    **fields: Any,
) -> None:
    logger.log(
        level,
        message or event,
        exc_info=exc_info,
        extra={
            "event": event,
            "structured_fields": {
                key: value
                for key, value in fields.items()
                if value is not None
            },
        },
    )
