"""Runtime-selectable, lazily loaded Laya inference backends."""

from __future__ import annotations

import importlib
import os
import platform
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol

BackendName = Literal["mlx", "torch", "onnx"]

BACKEND_ENV = "LAYA_MCP_BACKEND"
MODEL_ENV = "LAYA_MCP_MODEL"
ONNX_MODEL_ENV = "LAYA_MCP_ONNX_MODEL"
ONNX_ARTIFACT_ENV = "LAYA_MCP_ONNX_ARTIFACT_PATH"

DEFAULT_MLX_MODEL = "aac6fef/laya-multilingual-mlx"
DEFAULT_TORCH_MODEL = "convaiinnovations/laya"
SUPPORTED_BACKENDS = ("auto", "mlx", "torch", "onnx")


class BackendConfigurationError(ValueError):
    """The selected backend does not have enough configuration to run."""


class BackendUnavailableError(RuntimeError):
    """The selected optional runtime is not installed or cannot run here."""


class Backend(Protocol):
    """Minimal common surface for every Laya runtime."""

    @property
    def name(self) -> str: ...

    @property
    def model(self) -> str: ...

    def load(self) -> Any: ...


@dataclass(frozen=True)
class BackendConfig:
    """Resolved backend settings, all available without importing a runtime."""

    requested: str
    name: BackendName
    model: str
    onnx_artifact_path: str | None = None

    def catalog(self) -> dict[str, str]:
        """Return serializable backend details for the MCP catalog."""
        details = {
            "requested": self.requested,
            "active": self.name,
            "model": self.model,
        }
        if self.onnx_artifact_path is not None:
            details["artifact_path"] = self.onnx_artifact_path
        return details


def _environment_value(environ: Mapping[str, str], name: str) -> str | None:
    value = environ.get(name)
    return value.strip() if isinstance(value, str) else None


def _is_apple_silicon(system: str, machine: str) -> bool:
    return system.lower() == "darwin" and machine.lower() in {"arm64", "aarch64"}


def resolve_backend_config(
    *,
    environ: Mapping[str, str] | None = None,
    system: str | None = None,
    machine: str | None = None,
    model_override: str | None = None,
) -> BackendConfig:
    """Resolve the selected backend and model without importing optional packages."""
    environ = os.environ if environ is None else environ
    requested = _environment_value(environ, BACKEND_ENV)
    requested = "auto" if requested is None else requested.lower()
    if requested not in SUPPORTED_BACKENDS:
        supported = "|".join(SUPPORTED_BACKENDS)
        raise BackendConfigurationError(
            f"{BACKEND_ENV} must be one of {supported}; received {requested!r}"
        )

    system = platform.system() if system is None else system
    machine = platform.machine() if machine is None else machine
    if requested == "auto":
        if _is_apple_silicon(system, machine):
            backend: BackendName = "mlx"
        elif system.lower() == "linux":
            backend = "torch"
        else:
            raise BackendConfigurationError(
                f"{BACKEND_ENV}=auto supports Apple Silicon macOS and Linux; "
                f"detected {system}/{machine}. Set {BACKEND_ENV} explicitly."
            )
    else:
        backend = requested  # type: ignore[assignment]

    configured_model = model_override or _environment_value(environ, MODEL_ENV)
    if backend == "mlx":
        model = configured_model or DEFAULT_MLX_MODEL
        return BackendConfig(requested=requested, name=backend, model=model)
    if backend == "torch":
        model = configured_model or DEFAULT_TORCH_MODEL
        return BackendConfig(requested=requested, name=backend, model=model)

    model = (
        _environment_value(environ, ONNX_MODEL_ENV)
        or configured_model
        or DEFAULT_TORCH_MODEL
    )
    artifact_path = _environment_value(environ, ONNX_ARTIFACT_ENV)
    if not artifact_path:
        raise BackendConfigurationError(
            f"{ONNX_ARTIFACT_ENV} must name an exported ONNX model when "
            f"{BACKEND_ENV}=onnx"
        )
    return BackendConfig(
        requested=requested,
        name=backend,
        model=model,
        onnx_artifact_path=artifact_path,
    )


def backend_metadata(
    *,
    environ: Mapping[str, str] | None = None,
    model_override: str | None = None,
) -> dict[str, str]:
    """Return the configured backend details without loading a model."""
    return resolve_backend_config(
        environ=environ,
        model_override=model_override,
    ).catalog()


class _BaseBackend:
    def __init__(self, config: BackendConfig) -> None:
        self.config = config

    @property
    def name(self) -> str:
        return self.config.name

    @property
    def model(self) -> str:
        return self.config.model


def _missing_backend(extra: str) -> BackendUnavailableError:
    return BackendUnavailableError(
        f"The {extra} backend is unavailable. Install it with "
        f"`uv sync --extra {extra}` or `pip install \"laya-mcp[{extra}]\"`, "
        f"then restart laya-mcp."
    )


class MlxBackend(_BaseBackend):
    """Apple Silicon runtime backed by the independently ported MLX package."""

    def load(self) -> Any:
        if not _is_apple_silicon(platform.system(), platform.machine()):
            raise BackendUnavailableError(
                "The mlx backend requires Apple Silicon macOS. Use "
                f"{BACKEND_ENV}=torch on Linux."
            )
        try:
            laya = importlib.import_module("laya_mlx")
            loader = laya.load
        except (ImportError, AttributeError) as error:
            raise _missing_backend("mlx") from error
        return loader(self.model)


class TorchBackend(_BaseBackend):
    """Official upstream Laya runtime backed by PyTorch."""

    def load(self) -> Any:
        try:
            laya = importlib.import_module("laya")
            loader = laya.load
        except (ImportError, AttributeError) as error:
            raise _missing_backend("torch") from error
        return loader(self.model)


class OnnxBackend(_BaseBackend):
    """Upstream ONNX Runtime adapter using an original checkpoint and exported graph."""

    def load(self) -> Any:
        artifact_path = self.config.onnx_artifact_path
        if artifact_path is None:
            raise BackendConfigurationError(
                f"{ONNX_ARTIFACT_ENV} must be configured for the onnx backend"
            )
        artifact = Path(artifact_path).expanduser()
        if not artifact.is_file():
            raise BackendConfigurationError(
                f"ONNX artifact not found at {str(artifact)!r}. Set "
                f"{ONNX_ARTIFACT_ENV} to an exported .onnx file."
            )
        try:
            module = importlib.import_module("laya.onnx_agent")
            agent_type = module.ONNXAgent
            return agent_type(self.model, str(artifact))
        except (ImportError, AttributeError) as error:
            raise _missing_backend("onnx") from error


def create_backend(
    config: BackendConfig | None = None,
    *,
    environ: Mapping[str, str] | None = None,
    model_override: str | None = None,
) -> Backend:
    """Build the configured backend without importing its optional runtime."""
    config = config or resolve_backend_config(
        environ=environ,
        model_override=model_override,
    )
    if config.name == "mlx":
        return MlxBackend(config)
    if config.name == "torch":
        return TorchBackend(config)
    return OnnxBackend(config)
