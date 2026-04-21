"""Load EvalCase instances from YAML case-definition files."""

from __future__ import annotations

import importlib.resources
import logging
import pathlib
import shlex
import warnings
from typing import TYPE_CHECKING, Any

import yaml

from agent_lemon_lime.evals.runner import EvalCase, EvalInput
from agent_lemon_lime.evals.standard import (
    EvalDomain,
    Evaluator,
    ExitCodeEvaluator,
    OutputContainsEvaluator,
)

if TYPE_CHECKING:
    from agent_lemon_lime.config import LemonConfig
    from agent_lemon_lime.harness.base import AbstractSandbox

from agent_lemon_lime.config import resolve_env

logger = logging.getLogger(__name__)


def load_cases_from_dir(
    directory: pathlib.Path | str,
    *,
    base_dir: pathlib.Path | None = None,
    run_command: list[str] | None = None,
    run_env: dict[str, str] | None = None,
) -> list[EvalCase]:
    """Return EvalCase objects from all *.yaml files in directory.

    When run_command is provided, prompt-only cases are auto-converted by
    appending --prompt <prompt> to the run_command. Without run_command,
    prompt-only cases emit a warning and are skipped.

    Returns an empty list if the directory does not exist.
    """
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
                directory, base_dir=project_dir, run_command=run_command, run_env=resolved,
            )
        )
    return cases


def load_cases_from_sandbox(
    config: LemonConfig,
    *,
    sandbox: AbstractSandbox,
) -> list[EvalCase]:
    """Load eval cases by reading YAML files from inside the sandbox.

    Used in image-only mode where eval directories exist inside the
    container image rather than on the local filesystem.
    """
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
                directory, listing.stderr.strip(),
            )
            continue
        for yaml_path in sorted(listing.stdout.strip().splitlines()):
            if not yaml_path:
                continue
            result = sandbox.exec(["cat", yaml_path])
            if result.exit_code != 0:
                logger.warning(
                    "Failed to read %s from sandbox: %s",
                    yaml_path, result.stderr.strip(),
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
        input=EvalInput(command=command, timeout_seconds=config.run.timeout_seconds, env=resolved),
        evaluators=[ExitCodeEvaluator()],
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
    from agent_lemon_lime.evals.standard import LLMJudgeEvaluator

    probes_pkg = importlib.resources.files("agent_lemon_lime.probes")
    cases: list[EvalCase] = []
    for resource in sorted(probes_pkg.iterdir(), key=lambda r: r.name):
        if not resource.name.endswith(".yaml"):
            continue
        text = resource.read_text(encoding="utf-8")
        parsed = _parse_case_content(text, run_command=run_command, run_env=run_env)
        cases.extend(parsed)

    if model:
        for case in cases:
            if case.judge_hint:
                case.evaluators.append(
                    LLMJudgeEvaluator(
                        judge_hint=case.judge_hint,
                        scp_yaml=scp_yaml,
                        config_yaml=config_yaml,
                        model=model,
                    )
                )
    return cases


def _parse_case_file(
    path: pathlib.Path,
    run_command: list[str] | None = None,
    run_env: dict[str, str] | None = None,
) -> list[EvalCase]:
    return _parse_case_content(
        path.read_text(), run_command=run_command, run_env=run_env,
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
    judge_hint = raw.get("judge_hint", "")
    domain_str = raw.get("domain", "correctness")
    try:
        domain = EvalDomain(domain_str)
    except ValueError:
        logger.warning("Unknown domain '%s' for case '%s' — using CORRECTNESS", domain_str, name)
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
    evaluators: list[Evaluator] = [ExitCodeEvaluator()]
    expected = raw.get("expected_output")
    if expected:
        evaluators.append(OutputContainsEvaluator(expected=str(expected)))
    env = dict(run_env) if run_env else {}
    return EvalCase(
        name=name,
        input=EvalInput(command=list(command), env=env),
        evaluators=evaluators,
        domain=domain,
        description=description,
        judge_hint=judge_hint,
    )
