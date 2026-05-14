"""Load EvalCase instances from YAML case-definition files."""

from __future__ import annotations

import importlib.resources
import logging
import pathlib
import shlex
import warnings
from typing import TYPE_CHECKING, Any

import yaml
from agent_eval.config import JudgeConfig

from agent_lemon_lime.evals.runner import EvalCase, EvalInput
from agent_lemon_lime.evals.standard import EvalDomain

if TYPE_CHECKING:
    from agent_lemon_lime.config import LemonConfig
    from agent_lemon_lime.harness.base import AbstractSandbox

from agent_lemon_lime.config import resolve_env

logger = logging.getLogger(__name__)

DEFAULT_EXIT_CHECK = JudgeConfig(
    name="exit-code",
    check='return outputs.get("exit_code", 1) == 0, "non-zero exit"',
)


def load_cases_from_dir(
    directory: pathlib.Path | str,
    *,
    base_dir: pathlib.Path | None = None,
    run_command: list[str] | None = None,
    run_env: dict[str, str] | None = None,
) -> list[EvalCase]:
    """Return EvalCase objects from all *.yaml files in directory."""
    d = pathlib.Path(directory)
    if base_dir is not None:
        d = base_dir / d
    if not d.exists():
        return []
    cases: list[EvalCase] = []
    for yaml_file in sorted(d.glob("*.yaml")):
        cases.extend(_parse_case_file(yaml_file, run_command=run_command, run_env=run_env))
    return cases


def load_cases_from_config(config: LemonConfig, *, project_dir: pathlib.Path) -> list[EvalCase]:
    """Load all cases from directories listed in config.evals.directories."""
    run_command = shlex.split(config.run.command)
    resolved = resolve_env(config.run.env) if config.run.env else {}
    cases: list[EvalCase] = []
    for directory in config.evals.directories:
        cases.extend(
            load_cases_from_dir(
                directory,
                base_dir=project_dir,
                run_command=run_command,
                run_env=resolved,
            )
        )
    return cases


def load_cases_from_sandbox(
    config: LemonConfig,
    *,
    sandbox: AbstractSandbox,
) -> list[EvalCase]:
    """Load eval cases by reading YAML files from inside the sandbox."""
    run_command = shlex.split(config.run.command)
    resolved = resolve_env(config.run.env) if config.run.env else {}
    cases: list[EvalCase] = []
    for directory in config.evals.directories:
        listing = sandbox.exec(
            ["find", directory, "-name", "*.yaml", "-type", "f"],
        )
        if listing.exit_code != 0:
            logger.warning(
                "Failed to list eval dir %s in sandbox: %s",
                directory,
                listing.stderr.strip(),
            )
            continue
        for yaml_path in sorted(listing.stdout.strip().splitlines()):
            if not yaml_path:
                continue
            result = sandbox.exec(["cat", yaml_path])
            if result.exit_code != 0:
                logger.warning(
                    "Failed to read %s from sandbox: %s",
                    yaml_path,
                    result.stderr.strip(),
                )
                continue
            cases.extend(
                _parse_case_content(
                    result.stdout,
                    run_command=run_command,
                    run_env=resolved,
                )
            )
    return cases


def default_case_from_config(config: LemonConfig) -> EvalCase:
    """Build a single smoke-test EvalCase from config.run.command."""
    command = shlex.split(config.run.command)
    resolved = resolve_env(config.run.env) if config.run.env else {}
    return EvalCase(
        name=f"{config.name}-runs",
        input=EvalInput(
            command=command,
            timeout_seconds=config.run.timeout_seconds,
            env=resolved,
        ),
        judges=[DEFAULT_EXIT_CHECK],
        domain=EvalDomain.CORRECTNESS,
        description=f"Smoke test: {config.run.command} exits 0",
    )


def load_builtin_probes(
    *,
    run_command: list[str],
    run_env: dict[str, str] | None = None,
    model: str | None = None,
    scp_yaml: str = "",
    config_yaml: str = "",
) -> list[EvalCase]:
    """Load built-in probe cases from the agent_lemon_lime.probes package."""
    probes_pkg = importlib.resources.files("agent_lemon_lime.probes")
    cases: list[EvalCase] = []
    for resource in sorted(probes_pkg.iterdir(), key=lambda r: r.name):
        if not resource.name.endswith(".yaml"):
            continue
        text = resource.read_text(encoding="utf-8")
        parsed = _parse_case_content(text, run_command=run_command, run_env=run_env)
        cases.extend(parsed)
    return cases


def _parse_case_file(
    path: pathlib.Path,
    run_command: list[str] | None = None,
    run_env: dict[str, str] | None = None,
) -> list[EvalCase]:
    return _parse_case_content(
        path.read_text(),
        run_command=run_command,
        run_env=run_env,
    )


def _parse_case_content(
    text: str,
    run_command: list[str] | None = None,
    run_env: dict[str, str] | None = None,
) -> list[EvalCase]:
    data = yaml.safe_load(text)
    if not isinstance(data, dict):
        return []
    return [
        c
        for raw in data.get("cases", [])
        if (c := _parse_case(raw, run_command=run_command, run_env=run_env)) is not None
    ]


def _parse_case(
    raw: dict[str, Any],
    run_command: list[str] | None = None,
    run_env: dict[str, str] | None = None,
) -> EvalCase | None:
    name = raw.get("name", "unnamed")
    description = raw.get("description", "")
    domain_str = raw.get("domain", "correctness")
    try:
        domain = EvalDomain(domain_str)
    except ValueError:
        logger.warning(
            "Unknown domain '%s' for case '%s' — using CORRECTNESS",
            domain_str,
            name,
        )
        domain = EvalDomain.CORRECTNESS

    inp = raw.get("input", {})
    command = inp.get("command")
    if command is None:
        prompt = inp.get("prompt")
        if prompt and run_command is not None:
            sanitized = " ".join(prompt.split())
            command = run_command + ["--prompt", sanitized]
        else:
            warnings.warn(
                f"Skipping case '{name}': no command and no run_command to derive one from.",
                stacklevel=3,
            )
            return None

    judges: list[JudgeConfig] = []

    raw_judges = raw.get("judges", [])
    for rj in raw_judges:
        judges.append(
            JudgeConfig(
                name=rj.get("name", ""),
                description=rj.get("description", ""),
                condition=rj.get("if", ""),
                check=rj.get("check", ""),
                prompt=rj.get("prompt", ""),
                prompt_file=rj.get("prompt_file", ""),
                context=rj.get("context", []),
                feedback_type=rj.get("feedback_type", ""),
                model=rj.get("model", ""),
                module=rj.get("module", ""),
                function=rj.get("function", ""),
            )
        )

    if not raw_judges:
        judges.append(DEFAULT_EXIT_CHECK)

    expected = raw.get("expected_output")
    if expected:
        escaped = str(expected).replace('"', '\\"')
        judges.append(
            JudgeConfig(
                name="output-contains",
                check=(
                    f'return "{escaped}" in outputs.get("stdout", ""), '
                    f'"expected \\"{escaped}\\" not found in stdout"'
                ),
            )
        )

    judge_hint = raw.get("judge_hint", "")
    if judge_hint:
        judges.append(JudgeConfig(name="behavioral", prompt=judge_hint))

    env = dict(run_env) if run_env else {}
    return EvalCase(
        name=name,
        input=EvalInput(command=list(command), env=env),
        judges=judges,
        domain=domain,
        description=description,
    )
