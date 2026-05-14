"""Tests for probe loading from builtin YAML files."""

from agent_lemon_lime.evals.loader import load_builtin_probes
from agent_lemon_lime.evals.standard import EvalDomain


def test_load_builtin_probes_returns_cases():
    cases = load_builtin_probes(run_command=["python", "agent.py"])
    assert len(cases) > 0


def test_load_builtin_probes_all_have_probe_prefix():
    cases = load_builtin_probes(run_command=["python", "agent.py"])
    for case in cases:
        assert case.name.startswith("probe-"), f"{case.name} missing probe- prefix"


def test_load_builtin_probes_expected_count():
    cases = load_builtin_probes(run_command=["python", "agent.py"])
    assert len(cases) == 21


def test_load_builtin_probes_domains_are_set():
    cases = load_builtin_probes(run_command=["python", "agent.py"])
    domains = {c.domain for c in cases}
    assert EvalDomain.SAFETY in domains
    assert EvalDomain.SECURITY in domains
    assert EvalDomain.BEHAVIORAL in domains
    assert EvalDomain.STABILITY in domains


def test_load_builtin_probes_have_behavioral_judge():
    """All probes should have a behavioral judge from their judges section."""
    cases = load_builtin_probes(run_command=["python", "agent.py"])
    for case in cases:
        behavioral_judges = [j for j in case.judges if j.name == "behavioral"]
        assert len(behavioral_judges) == 1, f"{case.name} missing behavioral judge"
        assert behavioral_judges[0].prompt, f"{case.name} has empty behavioral prompt"


def test_load_builtin_probes_commands_use_run_command():
    cases = load_builtin_probes(run_command=["python", "my_agent.py"])
    for case in cases:
        assert case.input.command[0] == "python"
        assert case.input.command[1] == "my_agent.py"
        assert "--prompt" in case.input.command


def test_load_builtin_probes_with_env():
    cases = load_builtin_probes(
        run_command=["python", "agent.py"],
        run_env={"MY_KEY": "val"},
    )
    for case in cases:
        assert case.input.env.get("MY_KEY") == "val"


def test_parse_case_respects_domain():
    cases = load_builtin_probes(run_command=["python", "agent.py"])
    boundary_cases = [c for c in cases if c.name.startswith("probe-boundary-")]
    for case in boundary_cases:
        assert case.domain == EvalDomain.SAFETY
    injection_cases = [c for c in cases if c.name.startswith("probe-injection-")]
    for case in injection_cases:
        assert case.domain == EvalDomain.SECURITY
