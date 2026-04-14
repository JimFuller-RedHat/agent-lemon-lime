"""Tests for Agent Lemon and Agent Lime."""

from agent_lemon_lime.agents.lemon import LemonAgent, LemonRunResult
from agent_lemon_lime.agents.lime import LimeAgent, LimeEvent, LimeEventType
from agent_lemon_lime.config import LemonConfig, RunMode
from agent_lemon_lime.evals.runner import EvalCase, EvalInput
from agent_lemon_lime.evals.standard import EvalDomain, ExitCodeEvaluator, OutputContainsEvaluator
from agent_lemon_lime.harness.mock import MockSandbox
from agent_lemon_lime.scp.models import SystemCapabilityProfile

MINIMAL_CONFIG = "name: test-agent\nrun:\n  command: echo hello\n"


def test_lemon_agent_creates():
    config = LemonConfig.from_yaml(MINIMAL_CONFIG)
    agent = LemonAgent(config=config, sandbox=MockSandbox())
    assert agent is not None


def test_lemon_discovery_returns_result():
    config = LemonConfig.from_yaml(MINIMAL_CONFIG)
    sandbox = MockSandbox()
    sandbox.register_command(["echo", "hello"], stdout="hello\n", exit_code=0)
    agent = LemonAgent(config=config, sandbox=sandbox)
    result = agent.run_discovery(eval_cases=[])
    assert isinstance(result, LemonRunResult)
    assert result.mode == RunMode.DISCOVERY
    assert result.scp is not None
    assert result.report is not None
    assert result.violations == []


def test_lemon_discovery_with_cases():
    config = LemonConfig.from_yaml(MINIMAL_CONFIG)
    sandbox = MockSandbox()
    sandbox.register_command(["echo", "hello"], stdout="hello\n", exit_code=0)
    cases = [
        EvalCase(
            name="echo-test",
            input=EvalInput(command=["echo", "hello"]),
            evaluators=[ExitCodeEvaluator(), OutputContainsEvaluator(expected="hello")],
            domain=EvalDomain.CORRECTNESS,
        )
    ]
    agent = LemonAgent(config=config, sandbox=sandbox)
    result = agent.run_discovery(eval_cases=cases)
    assert result.report.summary.total == 1
    assert result.report.summary.passed == 1


def test_lemon_assert_no_violations():
    config = LemonConfig.from_yaml(MINIMAL_CONFIG)
    sandbox = MockSandbox()
    # SCP allows everything (empty = no restrictions asserted)
    allowed_scp = SystemCapabilityProfile()
    agent = LemonAgent(config=config, sandbox=sandbox)
    result = agent.run_assert(eval_cases=[], assert_scp=allowed_scp)
    assert result.mode == RunMode.ASSERT
    assert result.violations == []


def test_lemon_assert_detects_violation():
    config = LemonConfig.from_yaml(MINIMAL_CONFIG)
    sandbox = MockSandbox()

    # The observed SCP has a network endpoint not in the allowed SCP
    allowed_scp = SystemCapabilityProfile()  # no network policies = nothing allowed

    # We'll inject an observed SCP that has a disallowed endpoint
    observed_scp = SystemCapabilityProfile.model_validate(
        {
            "network_policies": {
                "external": {
                    "name": "External",
                    "endpoints": [{"host": "evil.example.com"}],
                }
            }
        }
    )

    agent = LemonAgent(config=config, sandbox=sandbox)
    # Override observed SCP for testing by injecting it
    result = agent.run_assert(
        eval_cases=[],
        assert_scp=allowed_scp,
        _observed_scp=observed_scp,
    )
    assert result.mode == RunMode.ASSERT
    assert len(result.violations) >= 1
    assert any("evil.example.com" in v for v in result.violations)


def test_lime_agent_creates():
    scp = SystemCapabilityProfile()
    lime = LimeAgent(otel_endpoint="http://localhost:4317", assert_scp=scp)
    assert lime.otel_endpoint == "http://localhost:4317"


def test_lime_analyse_events_no_violations():
    scp = SystemCapabilityProfile.model_validate(
        {
            "network_policies": {
                "anthropic": {
                    "name": "Anthropic",
                    "endpoints": [{"host": "api.anthropic.com"}],
                }
            }
        }
    )
    lime = LimeAgent(otel_endpoint="http://localhost:4317", assert_scp=scp)
    events = [
        LimeEvent(event_type=LimeEventType.TOOL_CALL, tool_name="file_read"),
    ]
    # file_read is not a network endpoint — no violations
    anomalies = lime.analyse_events(events)
    assert anomalies == []


def test_lime_analyse_events_detects_unknown_network_call():
    """An event that signals an unexpected outbound host is a violation."""
    scp = SystemCapabilityProfile()  # no network policies = nothing allowed
    lime = LimeAgent(otel_endpoint="http://localhost:4317", assert_scp=scp)
    events = [
        LimeEvent(
            event_type=LimeEventType.NETWORK_CALL,
            metadata={"host": "evil.example.com", "port": 443},
        )
    ]
    anomalies = lime.analyse_events(events)
    assert len(anomalies) == 1
    assert "evil.example.com" in anomalies[0]


def test_lime_collect_events_returns_list():
    """collect_events_from_otel returns a list (empty if collector unreachable)."""
    scp = SystemCapabilityProfile()
    lime = LimeAgent(otel_endpoint="http://localhost:4317", assert_scp=scp)
    events = lime.collect_events_from_otel()
    assert isinstance(events, list)
