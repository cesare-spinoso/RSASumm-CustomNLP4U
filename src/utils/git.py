from pathlib import Path

from sh import git


def commit(cwd: Path, msg: str) -> None:
    """Commit all files under the cwd with message msg."""
    assert isinstance(cwd, Path)
    assert isinstance(msg, str)
    git.add(cwd)
    git.commit("-m", msg)
