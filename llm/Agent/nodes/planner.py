import json

from pydantic import ValidationError

from llm.Agent.nodes.universal import _available_tools, _chat_completion, add_log
from llm.Agent.prompt import PLANNER_PROMPT
from llm.Agent.state import (
    AgentFailure,
    AgentPlan,
    AgentState,
    MAX_PLAN_STEPS,
    PlannerMode,
    PlanStep,
    PlanStepState,
    PlanUpdate,
)


def planner_node(state: AgentState) -> AgentState:
    try:
        question = state.get("question")
        if not question or not question.strip():
            raise ValueError("question cannot be empty")
        planner_mode = _planner_mode(state)
        payload = _planner_payload(state=state, question=question, planner_mode=planner_mode)
        content = _chat_completion(
            system_prompt=PLANNER_PROMPT,
            user_message=json.dumps(payload, ensure_ascii=False),
            response_format={"type": "json_object"},
        )
        agent_plan = _parse_agent_plan(content)
        org_planstate = _planned_steps_for_mode(
            state=state,
            planner_mode=planner_mode,
            steps=agent_plan.steps,
        )

        current_revision = int(state.get("plan_revision", 0) or 0)
        is_replan = planner_mode != "initial" or current_revision > 0
        next_revision = current_revision + 1 if is_replan else 1
        plan_updates = list(state.get("plan_updates", []))

        if is_replan:
            plan_updates.append(
                PlanUpdate(
                    revision=next_revision,
                    reason=agent_plan.reason or f"Plan replanned by {planner_mode}.",
                    changed_steps=_changed_steps_for_mode(
                        state=state,
                        planner_mode=planner_mode,
                        plan=org_planstate,
                    ),
                ).model_dump()
            )

        update: AgentState = {
            "plan": org_planstate,
            "plan_revision": next_revision,
            "plan_updates": plan_updates,
            "planner_mode": "initial",
            "react_results": [],
            "phase": "replanning" if is_replan else "planning",
            "agent_status": "running",
            "logs": add_log(
                state=state,
                node="planner_node",
                message=f"{planner_mode} plan created" if not is_replan else f"{planner_mode} plan replanned",
                extra={
                    "planner_mode": planner_mode,
                    "plan_revision": next_revision,
                    "step_count": len(org_planstate),
                },
            ),
        }
        if planner_mode == "step_replan":
            current_step_index = _step_replan_start_index(state)
            update["current_step_index"] = current_step_index
            update["current_step_id"] = org_planstate[current_step_index]["step_id"]
            update["no_finding_counts"] = {}
        elif planner_mode == "replan":
            update["current_step_index"] = 0
            update["current_step_id"] = org_planstate[0]["step_id"]
            update["no_finding_counts"] = {}
        return update
    except Exception as exc:
        return _planner_failed(state, str(exc))


def _planner_mode(state: AgentState) -> PlannerMode:
    value = state.get("planner_mode")
    if value is None:
        return "replan" if int(state.get("plan_revision", 0) or 0) > 0 else "initial"
    if value not in {"initial", "replan", "step_replan"}:
        raise ValueError(f"unknown planner_mode: {value}")
    return value


def _planner_payload(
    state: AgentState,
    question: str,
    planner_mode: PlannerMode,
) -> dict[str, object]:
    replan_context = state.get("replan_context", {})
    payload: dict[str, object] = {
        "planner_mode": planner_mode,
        "question": question,
        "document_id": state.get("document_id"),
        "available_tools": _available_tools(),
    }
    if planner_mode in {"replan", "step_replan"}:
        payload["plan"] = state.get("plan", [])
        payload["completed_steps"] = state.get("step_results", [])
    if planner_mode == "replan":
        payload["last_tool_observation"] = state.get("last_tool_observation") or replan_context.get(
            "last_tool_observation"
        )
        payload["react_results"] = replan_context.get("react_results", state.get("react_results", []))
    if planner_mode == "step_replan":
        payload["current_step"] = replan_context.get("current_step") or _current_step_payload(state)
        payload["react_results"] = replan_context.get("react_results", state.get("react_results", []))
        payload["no_finding_count"] = replan_context.get(
            "no_finding_count",
            _current_no_finding_count(state),
        )
    return payload


def _planned_steps_for_mode(
    state: AgentState,
    planner_mode: PlannerMode,
    steps: list[PlanStep],
) -> list[PlanStepState]:
    if planner_mode == "step_replan":
        start_index = _step_replan_start_index(state)
        preserved_steps = list(state.get("plan", []))[:start_index]
        replacement_steps = _restore_plan_content(
            steps,
            start_index=start_index + 1,
            force_step_ids=True,
        )
        return preserved_steps + replacement_steps
    return _restore_plan_content(steps)


def _changed_steps_for_mode(
    state: AgentState,
    planner_mode: PlannerMode,
    plan: list[PlanStepState],
) -> list[str]:
    if planner_mode == "step_replan":
        return [step["step_id"] for step in plan[_step_replan_start_index(state) :]]
    return [step["step_id"] for step in plan]


def _step_replan_start_index(state: AgentState) -> int:
    plan = state.get("plan", [])
    current_step_index = state.get("current_step_index")
    if isinstance(current_step_index, int) and 0 <= current_step_index < len(plan):
        return current_step_index

    current_step_id = state.get("current_step_id")
    if current_step_id:
        for index, step in enumerate(plan):
            if step.get("step_id") == current_step_id:
                return index
    raise ValueError("current step is required for step_replan")


def _current_step_payload(state: AgentState) -> dict[str, object]:
    plan = state.get("plan", [])
    return dict(plan[_step_replan_start_index(state)])


def _current_no_finding_count(state: AgentState) -> int:
    current_step_id = state.get("current_step_id")
    if not current_step_id:
        return 0
    return int(state.get("no_finding_counts", {}).get(current_step_id, 0) or 0)


def _parse_agent_plan(content: str) -> AgentPlan:
    try:
        data = json.loads(content)
    except json.JSONDecodeError as exc:
        raise ValueError("planner returned invalid JSON") from exc

    if not isinstance(data, dict):
        raise ValueError("planner returned non-object JSON")

    steps = data.get("steps")
    if not isinstance(steps, list):
        raise ValueError("planner response must include steps list")
    if not steps:
        raise ValueError("planner returned empty steps")
    if len(steps) > MAX_PLAN_STEPS:
        raise ValueError(f"planner returned more than {MAX_PLAN_STEPS} steps")

    try:
        return AgentPlan(
            steps=[PlanStep(**step) for step in steps],
            reason=data.get("reason"),
        )
    except (TypeError, ValidationError) as exc:
        raise ValueError("planner returned invalid plan steps") from exc


def _restore_plan_content(
    steps: list[PlanStep],
    *,
    start_index: int = 1,
    force_step_ids: bool = False,
) -> list[PlanStepState]:
    restored_steps: list[PlanStepState] = []

    for index, step in enumerate(steps, start=start_index):
        restored_step = PlanStep(
            step_id=f"step_{index}" if force_step_ids else step.step_id or f"step_{index}",
            task=step.task,
        )
        restored_steps.append(restored_step.model_dump())

    return restored_steps


def _planner_failed(state: AgentState, error: str) -> AgentState:
    message = f"planner failed: {error}"
    failure = AgentFailure(
        reason="planner_failed",
        message=message,
        node="planner_node",
        target_step_id=None,
    )
    return {
        "phase": "failed",
        "agent_status": "failed",
        "error": message,
        "failure": failure.model_dump(),
        "logs": add_log(
            state=state,
            node="planner_node",
            message="planner failed",
            extra={"error": error},
        ),
    }
