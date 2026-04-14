"""SkillLoader — loads evaluation skills from local directories and remote git repos."""
from __future__ import annotations

import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class Skill:
    name: str
    description: str
    commands: list[str] = field(default_factory=list)
    source: str = "local"

    @classmethod
    def from_yaml(cls, path: Path) -> Skill:
        data = yaml.safe_load(path.read_text())
        return cls(
            name=data["name"],
            description=data.get("description", ""),
            commands=data.get("commands", []),
        )


class SkillLoader:
    """Loads Skill definitions from local paths or remote git repositories."""

    def load_local(self, directory: Path | str) -> list[Skill]:
        """Load all *.yaml skill files from a local directory (non-recursive)."""
        d = Path(directory)
        skills = []
        for yaml_file in sorted(d.glob("*.yaml")):
            try:
                skill = Skill.from_yaml(yaml_file)
                skill.source = str(yaml_file)
                skills.append(skill)
            except (KeyError, yaml.YAMLError):
                # Skip malformed skill files
                continue
        return skills

    def load_from_repo(
        self, repo_url: str, subdirectory: str = "", branch: str = "main"
    ) -> list[Skill]:
        """Clone a git repo to a temp dir and load skills from it."""
        with tempfile.TemporaryDirectory() as tmp:
            subprocess.run(
                ["git", "clone", "--depth=1", "--branch", branch, repo_url, tmp],
                check=True,
                capture_output=True,
            )
            skill_dir = Path(tmp) / subdirectory if subdirectory else Path(tmp)
            skills = self.load_local(skill_dir)
            for skill in skills:
                skill.source = f"{repo_url}#{branch}/{subdirectory}"
            return skills
