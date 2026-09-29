"""Tests for lazy Laya backend selection and loading."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from laya_mcp import backends


@pytest.mark.parametrize(
    ("system", "machine", "backend_name", "model"),
    (
        (
            "Darwin",
            "arm64",
            "mlx",
            "aac6fef/laya-multilingual-mlx",
        ),
        (
            "Linux",
            "x86_64",
            "torch",
            "convaiinnovations/laya-multilingual",
        ),
    ),
)
def test_auto_backend_uses_platform_specific_defaults(
    system: str,
    machine: str,
    backend_name: str,
    model: str,
) -> None:
    """Auto selection needs no runtime import to identify the right backend."""
    config = backends.resolve_backend_config(
        environ={},
        system=system,
        machine=machine,
    )

    assert config.requested == "auto"
    assert config.name == backend_name
    assert config.model == model


def test_backend_override_and_model_override_are_respected() -> None:
    """An operator can select a runtime and checkpoint explicitly."""
    config = backends.resolve_backend_config(
        environ={
            "LAYA_MCP_BACKEND": "torch",
            "LAYA_MCP_MODEL": "example/custom-laya",
        },
        system="Darwin",
        machine="arm64",
    )

    assert config.requested == "torch"
    assert config.name == "torch"
    assert config.model == "example/custom-laya"


def test_onnx_requires_an_explicit_artifact_and_retains_original_model() -> None:
    """ONNX needs both tokenizer/config provenance and a local graph."""
    with pytest.raises(
        backends.BackendConfigurationError,
        match="LAYA_MCP_ONNX_ARTIFACT_PATH",
    ):
        backends.resolve_backend_config(
            environ={"LAYA_MCP_BACKEND": "onnx"},
            system="Linux",
            machine="x86_64",
        )

    config = backends.resolve_backend_config(
        environ={
            "LAYA_MCP_BACKEND": "onnx",
            "LAYA_MCP_ONNX_MODEL": "example/original-laya",
            "LAYA_MCP_ONNX_ARTIFACT_PATH": "/models/laya.onnx",
        },
        system="Linux",
        machine="x86_64",
    )

    assert config.name == "onnx"
    assert config.model == "example/original-laya"
    assert config.onnx_artifact_path == "/models/laya.onnx"


def test_unsupported_auto_platform_has_an_actionable_error() -> None:
    """A host with no default cannot silently select an incompatible runtime."""
    with pytest.raises(
        backends.BackendConfigurationError,
        match="Set LAYA_MCP_BACKEND explicitly",
    ):
        backends.resolve_backend_config(
            environ={},
            system="Windows",
            machine="AMD64",
        )


def test_invalid_backend_name_has_an_actionable_error() -> None:
    """Typographical errors do not fail later as opaque import errors."""
    with pytest.raises(
        backends.BackendConfigurationError,
        match="auto\\|mlx\\|torch\\|onnx",
    ):
        backends.resolve_backend_config(
            environ={"LAYA_MCP_BACKEND": "invalid"},
            system="Linux",
            machine="x86_64",
        )


def test_torch_runtime_is_imported_only_on_first_load(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Server startup can inspect configuration without importing PyTorch or Laya."""
    imported: list[str] = []
    loaded: list[str] = []
    agent = object()

    def load(model: str) -> object:
        loaded.append(model)
        return agent

    def import_module(name: str) -> SimpleNamespace:
        imported.append(name)
        return SimpleNamespace(load=load)

    monkeypatch.setattr(backends.importlib, "import_module", import_module)
    backend = backends.create_backend(
        backends.BackendConfig(
            requested="torch",
            name="torch",
            model="example/laya",
        )
    )

    assert imported == []
    assert backend.load() is agent
    assert imported == ["laya"]
    assert loaded == ["example/laya"]


def test_missing_torch_runtime_explains_how_to_install_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Optional dependencies fail at inference with the project's install command."""
    def import_module(_name: str) -> SimpleNamespace:
        raise ModuleNotFoundError("No module named 'laya'", name="laya")

    monkeypatch.setattr(backends.importlib, "import_module", import_module)
    backend = backends.TorchBackend(
        backends.BackendConfig(
            requested="torch",
            name="torch",
            model="example/laya",
        )
    )

    with pytest.raises(backends.BackendUnavailableError, match="uv sync --extra torch"):
        backend.load()


def test_onnx_runtime_receives_original_checkpoint_and_graph(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The future backend uses upstream's ONNX agent through the common interface."""
    artifact = tmp_path / "laya.onnx"
    artifact.touch()
    captured: dict[str, str] = {}

    class FakeOnnxAgent:
        def __init__(self, model: str, onnx_path: str) -> None:
            captured["model"] = model
            captured["onnx_path"] = onnx_path

    def import_module(name: str) -> SimpleNamespace:
        assert name == "laya.onnx_agent"
        return SimpleNamespace(ONNXAgent=FakeOnnxAgent)

    monkeypatch.setattr(backends.importlib, "import_module", import_module)
    backend = backends.OnnxBackend(
        backends.BackendConfig(
            requested="onnx",
            name="onnx",
            model="example/original-laya",
            onnx_artifact_path=str(artifact),
        )
    )

    assert isinstance(backend.load(), FakeOnnxAgent)
    assert captured == {
        "model": "example/original-laya",
        "onnx_path": str(artifact),
    }
