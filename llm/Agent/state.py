# llm/Agent/state.py

from typing import Any, Literal, TypedDict

from pydantic import BaseModel, Field


StepStatus = Literal["pending", "running", "done", "failed"]
AgentStatus = Literal["running", "finished", "failed"]
AgentPhase = Literal[
    "planning",
    "selecting_step",
    "reacting",
    "step_reflecting",
    "updating_plan",
    "replanning",
    "finalizing",
    "final_reflecting",
    "rewriting_final",
    "agent_gating",
    "finished",
    "failed",
]
ReflectionScope = Literal["step", "plan", "final"]
ReflectionStatus = Literal["pass", "retry_react", "replan", "rewrite_final", "fail"]
ReflectionSeverity = Literal["low", "medium", "high"]
StepReflectionRoute = Literal["pass", "retry_react", "replan", "fail"]
ContinueRoute = Literal["continue", "finish"]
FinalReflectionRoute = Literal[
    "pass",
    "rewrite_final",
    "retry_react",
    "replan",
    "fail",
]
AgentGateStatus = Literal["pass", "fail"]
FailureReason = Literal[
    "planner_failed",
    "react_failed",
    "reflection_failed",
    "retry_limit_exceeded",
    "final_gate_failed",
    "unknown",
]

MAX_PLAN_STEPS = 8
MAX_REPLAN_COUNT = 2
MAX_REACT_TURNS_PER_STEP = 4
MAX_REACT_RETRY_COUNT = 2
MAX_FINAL_REWRITE_COUNT = 1


class PlanStep(BaseModel):
    step_id: str = Field(..., description="Step id, such as step_1")
    task: str = Field(..., description="What this step should do")
    status: StepStatus = "pending"
    result: str | None = None
    retry_count: int = 0


class AgentPlan(BaseModel):
    steps: list[PlanStep]
    revision: int = 1
    reason: str | None = None


class ReactResult(BaseModel):
    step_id: str
    thought: str
    need: str
    action: str
    action_input: dict[str, Any] = Field(default_factory=dict)
    observation: str
    result: str
    success: bool


class ReflectionDecision(BaseModel):
    scope: ReflectionScope
    status: ReflectionStatus
    severity: ReflectionSeverity
    target_step_id: str | None = None
    problem: str = ""
    correction_instruction: str | None = None


class PlanUpdate(BaseModel):
    revision: int
    reason: str
    changed_steps: list[str] = Field(default_factory=list)


class FinalGateDecision(BaseModel):
    status: AgentGateStatus
    problem: str = ""
    correction_instruction: str | None = None


class AgentFailure(BaseModel):
    reason: FailureReason
    message: str
    node: str | None = None
    target_step_id: str | None = None


class AgentState(TypedDict, total=False):
    question: str
    document_id: str | None

    plan: list[dict[str, Any]]
    plan_revision: int
    plan_updates: list[dict[str, Any]]
    current_step_index: int
    current_step_id: str | None

    current_react_trace: list[dict[str, Any]]
    current_react_turn_count: int
    current_correction_instruction: str | None
    step_results: list[dict[str, Any]]
    react_results: list[dict[str, Any]]
    step_retry_counts: dict[str, int]

    reflection_decisions: list[dict[str, Any]]
    step_reflection_next: StepReflectionRoute
    should_continue_next: ContinueRoute
    final_reflection_next: FinalReflectionRoute
    replan_count: int
    react_retry_count: int
    final_rewrite_count: int

    draft_final_answer: str
    final_answer: str
    final_gate_decision: dict[str, Any] | None
    phase: AgentPhase
    agent_status: AgentStatus

    error: str | None
    failure: dict[str, Any] | None
    logs: list[dict[str, Any]]
