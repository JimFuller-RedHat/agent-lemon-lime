"""SkillLoader: loads eval skills from local directories or remote git repos."""

from __future__ import annotations

import pathlib
import subprocess
import tempfile
from dataclasses import dataclass


@dataclass
class Skill:
    name: str
    content: str
    source: str  # file path or git URL


class SkillLoader:
    """Load markdown skills from local dirs and remote git repos."""

    def load_from_dir(self, directory: pathlib.Path | str) -> list[Skill]:
        """Load all markdown files from a local directory tree.

        Args:
            directory: Directory to search recursively for .md files.

        Raises:
            FileNotFoundError: If the directory does not exist.
        """
        p = pathlib.Path(directory)
        if not p.exists():
            raise FileNotFoundError(f"Skill directory not found: {p}")
        return [
            Skill(name=md.stem, content=md.read_text(), source=str(md))
            for md in sorted(p.glob("**/*.md"))
        ]

    def load_from_git(
        self,
        url: str,
        *,
        branch: str = "main",
        subdirectory: str | None = None,
    ) -> list[Skill]:
        """Clone a git repo and load markdown skills from it.

        Args:
            url: Git repository URL to clone.
            branch: Branch to check out (default: main).
            subdirectory: Optional subdirectory within the repo to load from.
        """
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = pathlib.Path(tmp)
            subprocess.run(
                ["git", "clone", "--depth", "1", "--branch", branch, url, str(tmp_path)],
                check=True,
                capture_output=True,
            )
            target = tmp_path / subdirectory if subdirectory else tmp_path
            return self.load_from_dir(target)

    def load_all(self, sources: list[dict[str, str]]) -> list[Skill]:
        """Load from a list of source dicts (from LemonConfig.evals.skills).

        Args:
            sources: List of dicts with either a 'path' or 'git' key.
        """
        all_skills: list[Skill] = []
        for source in sources:
            if source.get("path"):
                all_skills.extend(self.load_from_dir(pathlib.Path(source["path"])))
            elif source.get("git"):
                all_skills.extend(
                    self.load_from_git(source["git"], branch=source.get("branch", "main"))
                )
        return all_skills
