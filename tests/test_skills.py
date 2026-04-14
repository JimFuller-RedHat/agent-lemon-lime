"""Tests for SkillLoader — local and remote skill loading."""
from pathlib import Path

from argus.core.skills import Skill, SkillLoader


def test_skill_from_yaml(tmp_path: Path) -> None:
    yaml_content = "name: test-skill\ndescription: A test skill\ncommands:\n  - echo hello\n"
    p = tmp_path / "skill.yaml"
    p.write_text(yaml_content)
    skill = Skill.from_yaml(p)
    assert skill.name == "test-skill"
    assert skill.commands == ["echo hello"]


def test_load_local_skills(tmp_path: Path) -> None:
    (tmp_path / "a.yaml").write_text("name: a\ndescription: A\ncommands: []\n")
    (tmp_path / "b.yaml").write_text("name: b\ndescription: B\ncommands: []\n")
    loader = SkillLoader()
    skills = loader.load_local(tmp_path)
    names = {s.name for s in skills}
    assert "a" in names and "b" in names


def test_load_local_ignores_non_yaml(tmp_path: Path) -> None:
    (tmp_path / "skill.yaml").write_text("name: s\ndescription: S\ncommands: []\n")
    (tmp_path / "readme.md").write_text("# docs")
    loader = SkillLoader()
    skills = loader.load_local(tmp_path)
    assert len(skills) == 1


def test_load_local_empty_dir(tmp_path: Path) -> None:
    loader = SkillLoader()
    skills = loader.load_local(tmp_path)
    assert skills == []
