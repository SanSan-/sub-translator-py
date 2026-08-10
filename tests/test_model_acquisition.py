import os
import traceback
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from sub_translate.models import TranslationOptions
from sub_translate.utils import huggingface as huggingface_utils
from sub_translate.utils.huggingface import (
    MODEL_MARKER_FILENAME,
    ModelAcquisitionError,
    acquire_local_model,
    inspect_model_directory,
)

MODEL_ID = "example/model"
MODEL_REVISION = "0123456789abcdef0123456789abcdef01234567"
REQUIRED_FILES = ("config.json", "tokenizer.json")
REQUIRED_FILE_GROUPS = (("model.safetensors", "model.safetensors.index.json"),)


def _write_complete_model(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "config.json").write_text("{}", encoding="utf-8")
    (path / "tokenizer.json").write_text("{}", encoding="utf-8")
    (path / "model.safetensors").write_bytes(b"weights")


def _write_snapshot(local_dir: Path, files: dict[str, bytes]) -> None:
    local_dir.mkdir(parents=True, exist_ok=True)
    for name, content in files.items():
        output_path = local_dir / name
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(content)


def _acquire(
    default_path: Path,
    *,
    model_path: Path | None = None,
    auto_download: bool = False,
    requires_hf_token: bool = False,
):
    return acquire_local_model(
        model_id=MODEL_ID,
        revision=MODEL_REVISION,
        model_path=model_path,
        default_model_path=default_path,
        auto_download=auto_download,
        required_files=REQUIRED_FILES,
        required_file_groups=REQUIRED_FILE_GROUPS,
        requires_hf_token=requires_hf_token,
    )


def test_translation_options_disable_model_download_by_default() -> None:
    options = TranslationOptions()

    assert options.model_path is None
    assert options.model_revision is None
    assert options.auto_download_model is False


def test_complete_explicit_directory_is_reused_without_download(tmp_path: Path) -> None:
    model_path = tmp_path / "selected"
    _write_complete_model(model_path)

    with patch("sub_translate.utils.huggingface._download_model_snapshot") as download:
        source = _acquire(tmp_path / "default", model_path=model_path)

    assert source.path == model_path.resolve()
    assert source.revision == MODEL_REVISION
    assert source.content_fingerprint is not None
    assert len(source.content_fingerprint) == 64
    assert source.revision_verified is False
    download.assert_not_called()


def test_complete_manual_model_with_auto_download_never_touches_hub(tmp_path: Path) -> None:
    model_path = tmp_path / "selected"
    _write_complete_model(model_path)

    with (
        patch("huggingface_hub.HfApi") as api_class,
        patch("huggingface_hub.snapshot_download") as snapshot_download,
        patch("sub_translate.utils.huggingface._download_model_snapshot") as download,
    ):
        source = _acquire(
            tmp_path / "default",
            model_path=model_path,
            auto_download=True,
        )

    assert source.path == model_path.resolve()
    assert source.revision_verified is False
    api_class.assert_not_called()
    snapshot_download.assert_not_called()
    download.assert_not_called()


def test_implicit_path_never_switches_to_huggingface_cache(tmp_path: Path) -> None:
    default_path = tmp_path / "default"
    cached_snapshot = tmp_path / "models--example--model" / "snapshots" / MODEL_REVISION
    _write_complete_model(cached_snapshot)

    with (
        patch("sub_translate.utils.huggingface._download_model_snapshot") as download,
        pytest.raises(ModelAcquisitionError, match="Каталог модели"),
    ):
        _acquire(default_path)

    download.assert_not_called()


def test_incomplete_directory_without_opt_in_fails_without_download(tmp_path: Path) -> None:
    model_path = tmp_path / "selected"
    model_path.mkdir()
    (model_path / "config.json").write_text("{}", encoding="utf-8")

    with (
        patch("sub_translate.utils.huggingface._download_model_snapshot") as download,
        pytest.raises(ModelAcquisitionError, match="auto_download_model"),
    ):
        _acquire(tmp_path / "default", model_path=model_path)

    download.assert_not_called()


def test_auto_download_uses_env_token_lock_space_check_and_same_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "selected"
    files = {
        "config.json": b"{}",
        "tokenizer.json": b"{}",
        "model.safetensors": b"weights",
    }
    model_info = SimpleNamespace(
        siblings=[SimpleNamespace(rfilename=name, size=len(content)) for name, content in files.items()]
    )
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setenv("HUGGINGFACE_TOKEN", "secret-from-env")

    def fake_snapshot_download(*_args, **kwargs) -> str:
        local_dir = Path(kwargs["local_dir"])
        _write_snapshot(local_dir, files)
        return str(local_dir)

    with (
        patch("huggingface_hub.HfApi") as api_class,
        patch("huggingface_hub.snapshot_download", side_effect=fake_snapshot_download) as download,
        patch("huggingface_hub.utils.WeakFileLock") as file_lock,
        patch(
            "sub_translate.utils.huggingface.shutil.disk_usage",
            return_value=SimpleNamespace(free=20 * 1024**3),
        ) as disk_usage,
    ):
        api_class.return_value.model_info.return_value = model_info
        source = _acquire(
            tmp_path / "default",
            model_path=target,
            auto_download=True,
            requires_hf_token=True,
        )

    assert source.path == target.resolve()
    assert source.revision == MODEL_REVISION
    api_class.assert_called_once_with(token="secret-from-env")
    api_class.return_value.model_info.assert_called_once_with(
        MODEL_ID,
        revision=MODEL_REVISION,
        files_metadata=True,
    )
    disk_usage.assert_called_once_with(target.parent.resolve())
    file_lock.assert_called_once()
    assert file_lock.call_args.kwargs["timeout"] > 0
    download.assert_called_once_with(
        repo_id=MODEL_ID,
        revision=MODEL_REVISION,
        local_dir=str(target.resolve()),
        force_download=False,
        token="secret-from-env",
    )
    marker = (target / ".sub_translate_model.json").read_text(encoding="utf-8")
    assert "secret-from-env" not in marker
    assert MODEL_REVISION in marker
    assert source.revision_verified is True
    assert source.content_fingerprint is not None

    with patch("sub_translate.utils.huggingface._download_model_snapshot") as second_download:
        reused = _acquire(
            tmp_path / "default",
            model_path=target,
            requires_hf_token=True,
        )
    assert reused == source
    second_download.assert_not_called()


def test_auto_download_fails_before_snapshot_when_space_is_insufficient(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("HF_TOKEN", raising=False)
    model_info = SimpleNamespace(siblings=[SimpleNamespace(rfilename="model.safetensors", size=10 * 1024**3)])

    with (
        patch("huggingface_hub.HfApi") as api_class,
        patch("huggingface_hub.snapshot_download") as download,
        patch("huggingface_hub.utils.WeakFileLock"),
        patch(
            "sub_translate.utils.huggingface.shutil.disk_usage",
            return_value=SimpleNamespace(free=1024),
        ),
    ):
        api_class.return_value.model_info.return_value = model_info
        with pytest.raises(ModelAcquisitionError, match="свободного места"):
            _acquire(tmp_path / "default", auto_download=True)

    download.assert_not_called()


def test_gated_download_requires_hf_token_before_hub_client(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGINGFACE_TOKEN", raising=False)

    with (
        patch("huggingface_hub.HfApi") as api_class,
        pytest.raises(ModelAcquisitionError, match="HF_TOKEN"),
    ):
        _acquire(
            tmp_path / "default",
            auto_download=True,
            requires_hf_token=True,
        )

    api_class.assert_not_called()


def test_huggingface_token_alias_is_env_only_and_hf_token_has_priority(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HF_TOKEN", "primary-token")
    monkeypatch.setenv("HUGGINGFACE_TOKEN", "fallback-token")

    assert huggingface_utils._huggingface_token_from_environment() == "primary-token"

    monkeypatch.delenv("HF_TOKEN")
    assert huggingface_utils._huggingface_token_from_environment() == "fallback-token"


def test_hub_error_does_not_expose_token_in_traceback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret = "sensitive-huggingface-token"
    default_path = tmp_path / "default"
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setenv("HUGGINGFACE_TOKEN", secret)

    with (
        patch("huggingface_hub.HfApi") as api_class,
        patch("huggingface_hub.utils.WeakFileLock"),
    ):
        api_class.return_value.model_info.side_effect = RuntimeError(secret)
        with pytest.raises(ModelAcquisitionError) as error:
            _acquire(
                default_path,
                auto_download=True,
                requires_hf_token=True,
            )

    rendered = "".join(traceback.format_exception(error.value))
    assert secret not in rendered


def test_auto_download_revalidates_required_files(tmp_path: Path) -> None:
    model_info = SimpleNamespace(siblings=[])

    with (
        patch("huggingface_hub.HfApi") as api_class,
        patch("huggingface_hub.snapshot_download", return_value=str(tmp_path / "default")),
        patch("huggingface_hub.utils.WeakFileLock"),
        patch(
            "sub_translate.utils.huggingface.shutil.disk_usage",
            return_value=SimpleNamespace(free=20 * 1024**3),
        ),
    ):
        api_class.return_value.model_info.return_value = model_info
        with pytest.raises(ModelAcquisitionError, match="после загрузки"):
            _acquire(tmp_path / "default", auto_download=True)


def test_auto_download_replaces_stale_marker_after_complete_snapshot(tmp_path: Path) -> None:
    target = tmp_path / "selected"
    _write_complete_model(target)
    (target / MODEL_MARKER_FILENAME).write_text(
        '{"model_id":"foreign/model","revision":"ffffffffffffffffffffffffffffffffffffffff"}',
        encoding="utf-8",
    )
    files = {
        "config.json": b"{}",
        "tokenizer.json": b"{}",
        "model.safetensors": b"weights",
    }
    model_info = SimpleNamespace(
        siblings=[SimpleNamespace(rfilename=name, size=len(content)) for name, content in files.items()]
    )

    def refresh_snapshot(*_args, **kwargs) -> str:
        local_dir = Path(kwargs["local_dir"])
        _write_snapshot(local_dir, files)
        return str(local_dir)

    (target / "model.safetensors").write_bytes(b"foreign")
    with (
        patch("huggingface_hub.HfApi") as api_class,
        patch("huggingface_hub.snapshot_download", side_effect=refresh_snapshot) as download,
        patch("huggingface_hub.utils.WeakFileLock"),
        patch(
            "sub_translate.utils.huggingface.shutil.disk_usage",
            return_value=SimpleNamespace(free=20 * 1024**3),
        ),
    ):
        api_class.return_value.model_info.return_value = model_info
        source = _acquire(
            tmp_path / "default",
            model_path=target,
            auto_download=True,
        )

    assert source.path == target.resolve()
    assert source.revision_verified is True
    assert download.call_args.kwargs["force_download"] is True
    assert Path(download.call_args.kwargs["local_dir"]) != target.resolve()
    assert (target / "model.safetensors").read_bytes() == files["model.safetensors"]
    marker = (target / MODEL_MARKER_FILENAME).read_text(encoding="utf-8")
    assert MODEL_ID in marker
    assert MODEL_REVISION in marker
    assert not tuple(tmp_path.glob(".selected.refresh-*"))
    assert not tuple(tmp_path.glob(".selected.backup-*"))


def test_incomplete_download_removes_files_absent_from_pinned_manifest(tmp_path: Path) -> None:
    target = tmp_path / "selected"
    target.mkdir()
    (target / "config.json").write_bytes(b"xx")
    (target / "model.safetensors").write_bytes(b"foreign-single")
    (target / "tokenizer_config.json").write_text('{"foreign":true}', encoding="utf-8")
    (target / "modeling_foreign.py").write_text("raise RuntimeError", encoding="utf-8")
    metadata_path = target / ".cache" / "huggingface" / "download" / "config.json.metadata"
    metadata_path.parent.mkdir(parents=True)
    metadata_path.write_text("resume metadata", encoding="utf-8")
    files = {
        "config.json": b"{}",
        "tokenizer.json": b"{}",
        "model.safetensors.index.json": (
            b'{"weight_map":{"layer.0":"model-00001-of-00002.safetensors",'
            b'"layer.1":"model-00002-of-00002.safetensors"}}'
        ),
        "model-00001-of-00002.safetensors": b"first",
        "model-00002-of-00002.safetensors": b"second",
    }
    model_info = SimpleNamespace(
        siblings=[SimpleNamespace(rfilename=name, size=len(content)) for name, content in files.items()]
    )

    def download_snapshot(*_args, **kwargs) -> str:
        local_dir = Path(kwargs["local_dir"])
        _write_snapshot(local_dir, files)
        return str(local_dir)

    with (
        patch("huggingface_hub.HfApi") as api_class,
        patch("huggingface_hub.snapshot_download", side_effect=download_snapshot) as download,
        patch("huggingface_hub.utils.WeakFileLock"),
        patch(
            "sub_translate.utils.huggingface.shutil.disk_usage",
            return_value=SimpleNamespace(free=20 * 1024**3),
        ),
    ):
        api_class.return_value.model_info.return_value = model_info
        source = _acquire(tmp_path / "default", model_path=target, auto_download=True)

    assert source.revision_verified is True
    assert download.call_args.kwargs["force_download"] is True
    assert Path(download.call_args.kwargs["local_dir"]) == target.resolve()
    assert (target / "config.json").read_bytes() == b"{}"
    assert not (target / "model.safetensors").exists()
    assert not (target / "tokenizer_config.json").exists()
    assert not (target / "modeling_foreign.py").exists()
    assert metadata_path.read_text(encoding="utf-8") == "resume metadata"
    assert (target / "model-00001-of-00002.safetensors").read_bytes() == b"first"
    assert (target / "model-00002-of-00002.safetensors").read_bytes() == b"second"


def test_failed_forced_refresh_preserves_target_and_removes_staging(tmp_path: Path) -> None:
    target = tmp_path / "selected"
    _write_complete_model(target)
    marker_path = target / MODEL_MARKER_FILENAME
    marker_path.write_text(
        '{"model_id":"foreign/model","revision":"ffffffffffffffffffffffffffffffffffffffff"}',
        encoding="utf-8",
    )
    original_files = {
        path.relative_to(target).as_posix(): path.read_bytes() for path in target.rglob("*") if path.is_file()
    }
    model_info = SimpleNamespace(siblings=[SimpleNamespace(rfilename="model.safetensors", size=len(b"replacement"))])

    with (
        patch("huggingface_hub.HfApi") as api_class,
        patch("huggingface_hub.snapshot_download", side_effect=RuntimeError("network failed")),
        patch("huggingface_hub.utils.WeakFileLock"),
        patch(
            "sub_translate.utils.huggingface.shutil.disk_usage",
            return_value=SimpleNamespace(free=20 * 1024**3),
        ),
        pytest.raises(ModelAcquisitionError, match="Не удалось загрузить"),
    ):
        api_class.return_value.model_info.return_value = model_info
        _acquire(tmp_path / "default", model_path=target, auto_download=True)

    actual_files = {
        path.relative_to(target).as_posix(): path.read_bytes() for path in target.rglob("*") if path.is_file()
    }
    assert actual_files == original_files
    assert not tuple(tmp_path.glob(".selected.refresh-*"))
    assert not tuple(tmp_path.glob(".selected.backup-*"))


@pytest.mark.parametrize("restore_fails", [False, True])
def test_failed_atomic_publish_preserves_original_target(tmp_path: Path, *, restore_fails: bool) -> None:
    target = tmp_path / "selected"
    _write_complete_model(target)
    marker_path = target / MODEL_MARKER_FILENAME
    marker_path.write_text(
        '{"model_id":"foreign/model","revision":"ffffffffffffffffffffffffffffffffffffffff"}',
        encoding="utf-8",
    )
    original_files = {
        path.relative_to(target).as_posix(): path.read_bytes() for path in target.rglob("*") if path.is_file()
    }
    files = {
        "config.json": b'{"replacement":true}',
        "tokenizer.json": b"{}",
        "model.safetensors": b"replacement",
    }
    model_info = SimpleNamespace(
        siblings=[SimpleNamespace(rfilename=name, size=len(content)) for name, content in files.items()]
    )
    real_replace = os.replace

    def fail_staging_publish(source: str | Path, destination: str | Path) -> None:
        source_path = Path(source)
        if source_path.name.startswith(".selected.refresh-") and Path(destination) == target.resolve():
            raise OSError("publish failed")
        if restore_fails and source_path.name.startswith(".selected.backup-"):
            raise OSError("restore failed")
        real_replace(source, destination)

    def download_snapshot(*_args, **kwargs) -> str:
        local_dir = Path(kwargs["local_dir"])
        _write_snapshot(local_dir, files)
        return str(local_dir)

    with (
        patch("huggingface_hub.HfApi") as api_class,
        patch("huggingface_hub.snapshot_download", side_effect=download_snapshot),
        patch("huggingface_hub.utils.WeakFileLock"),
        patch(
            "sub_translate.utils.huggingface.shutil.disk_usage",
            return_value=SimpleNamespace(free=20 * 1024**3),
        ),
        patch("sub_translate.utils.huggingface.os.replace", side_effect=fail_staging_publish),
        pytest.raises(
            ModelAcquisitionError,
            match="исходный каталог" if restore_fails else "атомарно заменить",
        ),
    ):
        api_class.return_value.model_info.return_value = model_info
        _acquire(tmp_path / "default", model_path=target, auto_download=True)

    preserved_path = next(iter(tmp_path.glob(".selected.backup-*")), target)
    actual_files = {
        path.relative_to(preserved_path).as_posix(): path.read_bytes()
        for path in preserved_path.rglob("*")
        if path.is_file()
    }
    assert actual_files == original_files
    assert not tuple(tmp_path.glob(".selected.refresh-*"))
    assert bool(tuple(tmp_path.glob(".selected.backup-*"))) is restore_fails


def test_auto_download_rejects_unpinned_revision_before_hub(tmp_path: Path) -> None:
    with (
        patch("huggingface_hub.HfApi") as api_class,
        pytest.raises(ModelAcquisitionError, match="40-символьная"),
    ):
        acquire_local_model(
            model_id=MODEL_ID,
            revision="main",
            model_path=None,
            default_model_path=tmp_path / "default",
            auto_download=True,
            required_files=REQUIRED_FILES,
            required_file_groups=REQUIRED_FILE_GROUPS,
        )

    api_class.assert_not_called()


def test_sharded_safetensors_index_requires_every_segment(tmp_path: Path) -> None:
    model_path = tmp_path / "sharded"
    model_path.mkdir()
    (model_path / "config.json").write_text("{}", encoding="utf-8")
    (model_path / "tokenizer.json").write_text("{}", encoding="utf-8")
    (model_path / "model.safetensors.index.json").write_text(
        '{"weight_map":{"layer.0":"model-00001-of-00002.safetensors","layer.1":"model-00002-of-00002.safetensors"}}',
        encoding="utf-8",
    )
    (model_path / "model-00001-of-00002.safetensors").write_bytes(b"first")

    incomplete = inspect_model_directory(
        model_path,
        model_id=MODEL_ID,
        revision=MODEL_REVISION,
        required_files=REQUIRED_FILES,
        required_file_groups=REQUIRED_FILE_GROUPS,
    )
    assert incomplete.complete is False
    assert incomplete.structurally_complete is False
    assert any("model-00002-of-00002.safetensors" in issue for issue in incomplete.issues)

    (model_path / "model-00002-of-00002.safetensors").write_bytes(b"second")
    complete = inspect_model_directory(
        model_path,
        model_id=MODEL_ID,
        revision=MODEL_REVISION,
        required_files=REQUIRED_FILES,
        required_file_groups=REQUIRED_FILE_GROUPS,
    )
    assert complete.complete is True
    assert complete.structurally_complete is True


def test_foreign_marker_makes_complete_model_unacceptable(tmp_path: Path) -> None:
    model_path = tmp_path / "selected"
    _write_complete_model(model_path)
    (model_path / MODEL_MARKER_FILENAME).write_text(
        '{"model_id":"example/model","revision":"ffffffffffffffffffffffffffffffffffffffff"}',
        encoding="utf-8",
    )

    inspection = inspect_model_directory(
        model_path,
        model_id=MODEL_ID,
        revision=MODEL_REVISION,
        required_files=REQUIRED_FILES,
        required_file_groups=REQUIRED_FILE_GROUPS,
    )
    with (
        patch("sub_translate.utils.huggingface._download_model_snapshot") as download,
        pytest.raises(ModelAcquisitionError, match="другой модели или ревизии"),
    ):
        _acquire(tmp_path / "default", model_path=model_path)

    assert inspection.complete is False
    assert inspection.structurally_complete is False
    assert inspection.revision_verified is False
    download.assert_not_called()


@pytest.mark.parametrize(
    "marker_content",
    [
        "{broken",
        "[]",
        '{"model_id":"example/model","revision":"0123456789abcdef0123456789abcdef01234567"}',
    ],
)
def test_damaged_marker_makes_complete_model_unacceptable(
    tmp_path: Path,
    marker_content: str,
) -> None:
    model_path = tmp_path / "selected"
    _write_complete_model(model_path)
    (model_path / MODEL_MARKER_FILENAME).write_text(marker_content, encoding="utf-8")

    inspection = inspect_model_directory(
        model_path,
        model_id=MODEL_ID,
        revision=MODEL_REVISION,
        required_files=REQUIRED_FILES,
        required_file_groups=REQUIRED_FILE_GROUPS,
    )

    assert inspection.complete is False
    assert inspection.structurally_complete is False
    assert any(MODEL_MARKER_FILENAME in issue for issue in inspection.issues)
    with pytest.raises(ModelAcquisitionError, match="повреждён"):
        _acquire(tmp_path / "default", model_path=model_path)


def test_changed_verified_marker_makes_model_unacceptable(tmp_path: Path) -> None:
    model_path = tmp_path / "selected"
    _write_complete_model(model_path)
    fingerprint = huggingface_utils.model_content_fingerprint(model_path)
    assert fingerprint is not None
    (model_path / MODEL_MARKER_FILENAME).write_text(
        (
            '{"model_id":"example/model","revision":"'
            + MODEL_REVISION
            + '","content_fingerprint":"'
            + fingerprint
            + '","fingerprint_version":2}'
        ),
        encoding="utf-8",
    )

    verified = _acquire(tmp_path / "default", model_path=model_path)
    assert verified.revision_verified is True

    (model_path / "model.safetensors").write_bytes(b"changed-weights")
    changed = inspect_model_directory(
        model_path,
        model_id=MODEL_ID,
        revision=MODEL_REVISION,
        required_files=REQUIRED_FILES,
        required_file_groups=REQUIRED_FILE_GROUPS,
    )
    assert changed.complete is False
    assert changed.structurally_complete is False
    assert changed.revision_verified is False
    with (
        patch("sub_translate.utils.huggingface._download_model_snapshot") as download,
        pytest.raises(ModelAcquisitionError, match="изменено после проверки"),
    ):
        _acquire(tmp_path / "default", model_path=model_path)
    download.assert_not_called()


def test_content_fingerprint_detects_middle_only_change(tmp_path: Path) -> None:
    model_path = tmp_path / "model"
    model_path.mkdir()
    weights = model_path / "model.safetensors"
    payload = bytearray(b"a" * (3 * 1024**2))
    weights.write_bytes(payload)
    first = huggingface_utils.model_content_fingerprint(model_path)
    first_stat = weights.stat()

    payload[len(payload) // 2] = ord("b")
    weights.write_bytes(payload)
    os.utime(
        weights,
        ns=(first_stat.st_atime_ns, first_stat.st_mtime_ns + 1_000_000_000),
    )
    second = huggingface_utils.model_content_fingerprint(model_path)

    assert first is not None
    assert second is not None
    assert second != first


def test_legacy_marker_remains_unverified_and_is_not_rewritten(tmp_path: Path) -> None:
    model_path = tmp_path / "selected"
    _write_complete_model(model_path)
    marker_path = model_path / MODEL_MARKER_FILENAME
    marker_path.write_text(
        (
            '{"model_id":"example/model","revision":"'
            + MODEL_REVISION
            + '","content_fingerprint":"'
            + ("a" * 64)
            + '"}'
        ),
        encoding="utf-8",
    )
    original_marker = marker_path.read_bytes()

    source = _acquire(tmp_path / "default", model_path=model_path)

    assert source.revision_verified is False
    assert source.content_fingerprint == huggingface_utils.model_content_fingerprint(model_path)
    assert marker_path.read_bytes() == original_marker


def test_lock_timeout_is_reported_as_acquisition_error(tmp_path: Path) -> None:
    with (
        patch(
            "huggingface_hub.utils.WeakFileLock",
            side_effect=TimeoutError("busy"),
        ),
        pytest.raises(ModelAcquisitionError, match="блокировку"),
    ):
        _acquire(tmp_path / "default", auto_download=True)
