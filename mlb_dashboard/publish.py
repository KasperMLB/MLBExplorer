from __future__ import annotations

import argparse
import inspect
from pathlib import Path

from .config import AppConfig

try:
    from huggingface_hub import HfApi
except ImportError:  # pragma: no cover
    HfApi = None


def publish_artifacts(artifacts_dir: Path, repo_id: str, repo_type: str, token: str | None = None) -> None:
    if HfApi is None:
        raise RuntimeError("huggingface_hub is required to publish artifacts.")
    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, repo_type=repo_type, exist_ok=True)
    large_folder_signature = inspect.signature(api.upload_large_folder)
    if "path_in_repo" in large_folder_signature.parameters:
        api.upload_large_folder(
            repo_id=repo_id,
            repo_type=repo_type,
            folder_path=str(artifacts_dir),
            path_in_repo=".",
        )
        return
    if hasattr(api, "upload_folder"):
        api.upload_folder(
            repo_id=repo_id,
            repo_type=repo_type,
            folder_path=str(artifacts_dir),
            path_in_repo=".",
        )
        return
    api.upload_large_folder(
        repo_id=repo_id,
        repo_type=repo_type,
        folder_path=str(artifacts_dir),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Publish hosted artifacts to Hugging Face.")
    parser.add_argument("--artifacts-dir", type=Path, required=True)
    parser.add_argument("--repo-id", type=str, required=True)
    parser.add_argument("--repo-type", type=str, default="dataset")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = AppConfig()
    publish_artifacts(
        artifacts_dir=args.artifacts_dir,
        repo_id=args.repo_id,
        repo_type=args.repo_type,
        token=config.hf_token or None,
    )


if __name__ == "__main__":
    main()
