# llm/Agent/state.py

from typing import Any, Literal, TypedDict

from pydantic import BaseModel, Field


StepStatus = Literal["pending", "running", "done", "failed"]
AgentStatus = Literal["running", "failed"]
AgentPhase = Literal[
    "planning",
    "selecting_step",
    "reacting",
    "replanning",
    "failed",
]
ContinueRoute = Literal["continue", "finish"]
AgentLoopDecisionType = Literal["think", "tool_call", "finish", "fail"]
AgentLoopSignal = Literal[
    "overthink",
    "tool_error",
    "overturning",
    "finding_missing",
]
PlannerMode = Literal["initial", "replan", "step_replan"]
FailureReason = Literal["planner_failed", "react_failed"]

MAX_PLAN_STEPS = 8
MAX_REPLAN_COUNT = 1
MAX_STEP_REPLAN_COUNT = 1
MAX_REACT_TURNS_PER_STEP = 7


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


class AgentLoopResult(BaseModel):
    thought: str
    decide_type: AgentLoopDecisionType
    Signal: AgentLoopSignal | None = None
    no_finding: int = Field(default=0, ge=0, le=1)
    tool_name: str | None = None
    arguments: dict[str, Any] = Field(default_factory=dict)
    observation: str | None = None
    answer: str = ""


class PlanUpdate(BaseModel):
    revision: int
    reason: str
    changed_steps: list[str] = Field(default_factory=list)


class AgentFailure(BaseModel):
    reason: FailureReason
    message: str
    node: str | None = None
    target_step_id: str | None = None


class PlanStepState(TypedDict):
    step_id: str
    task: str
    status: StepStatus
    result: str | None
    retry_count: int


class AgentState(TypedDict, total=False):
    question: str
    document_id: str | None

    plan: list[PlanStepState]
    plan_revision: int
    plan_updates: list[dict[str, Any]]
    current_step_index: int
    current_step_id: str | None

    planner_mode: PlannerMode
    replan_context: dict[str, Any]
    last_tool_observation: str | None
    current_react_turn_count: int
    current_correction_instruction: str | None
    step_results: list[dict[str, Any]]
    react_results: list[dict[str, Any]]
    tool_calls: list[str]
    failed_tools: list[str]
    overthink_counts: dict[str, int]
    no_finding_counts: dict[str, int]
    agent_depth: int
    subagent_results: list[dict[str, Any]]

    should_continue_next: ContinueRoute
    replan_count: int
    step_replan_count: int

    phase: AgentPhase
    agent_status: AgentStatus

    error: str | None
    failure: dict[str, Any] | None
    logs: list[dict[str, Any]]
