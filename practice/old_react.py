"""
import json
from typing import Any

from pydantic import ValidationError

from llm.Agent.nodes.universal import _available_tools, _chat_completion, add_log
from llm.Agent.prompt import (
    REACT_ACTING_PROMPT,
    REACT_REASONING_PROMPT,
    REACT_RESULT_SUMMARY_PROMPT,
)
from llm.Agent.state import (
    AgentFailure,
    AgentState,
    MAX_REACT_TURNS_PER_STEP,
    PlanStep,
    PlanStepState,
    ReactResult,
)
from llm.tools import TOOL_REGISTRY


def react_node(state: AgentState) -> AgentState:
    try:
        current_step_index, current_step = _current_step(state)
        question = state.get("question", "")
        step_id = current_step["step_id"]
        task = current_step["task"]
        current_react_trace = list(state.get("current_react_trace", []))
        react_results = list(state.get("react_results", []))
        step_results = list(state.get("step_results", []))

        for turn_index in range(MAX_REACT_TURNS_PER_STEP):
            reasoning = _reason_about_next_turn(
                state=state,
                question=question,
                step_id=step_id,
                task=task,
                current_react_trace=current_react_trace,
            )
            acting = _choose_action(
                question=question,
                step_id=step_id,
                task=task,
                current_react_trace=current_react_trace,
                reasoning=reasoning,
            )
            observation = _execute_action(
                action=acting["action"],
                action_input=acting["action_input"],
                document_id=state.get("document_id"),
            )
            result_summary = _summarize_turn_result(
                question=question,
                step_id=step_id,
                task=task,
                current_react_trace=current_react_trace,
                reasoning=reasoning,
                acting=acting,
                observation=observation,
            )
            react_result = ReactResult(
                step_id=step_id,
                thought=reasoning["thought"],
                need=reasoning["need"],
                action=acting["action"],
                action_input=acting["action_input"],
                observation=observation,
                result=result_summary["result"],
                success=result_summary["success"],
            ).model_dump()
            current_react_trace.append(react_result)
            react_results.append(react_result)

            if reasoning["done"]:
                updated_plan = _update_step_result(
                    plan=list(state.get("plan", [])),
                    current_step_index=current_step_index,
                    result=result_summary["result"],
                )
                step_results.append(
                    {
                        "step_id": step_id,
                        "task": task,
                        "result": result_summary["result"],
                    }
                )
                return {
                    "plan": updated_plan,
                    "current_react_trace": current_react_trace,
                    "current_react_turn_count": turn_index + 1,
                    "react_results": react_results,
                    "tool_calls": _tool_calls_from_react_results(react_results),
                    "step_results": step_results,
                    "phase": "reacting",
                    "agent_status": "running",
                    "logs": add_log(
                        state=state,
                        node="react_node",
                        message="step react completed",
                        extra={
                            "current_step_id": step_id,
                            "react_turn_count": turn_index + 1,
                        },
                    ),
                }

        raise ValueError(
            f"react exceeded {MAX_REACT_TURNS_PER_STEP} turns before step completed"
        )
    except Exception as exc:
        return _react_failed(state, str(exc))


def _current_step(state: AgentState) -> tuple[int, PlanStepState]:
    plan = state.get("plan", [])
    current_step_index = state.get("current_step_index")
    current_step_id = state.get("current_step_id")

    if current_step_index is None:
        raise ValueError("current_step_index is required")
    if not isinstance(current_step_index, int):
        raise ValueError("current_step_index must be an integer")
    if current_step_index < 0 or current_step_index >= len(plan):
        raise ValueError("current_step_index is out of range")
    if not current_step_id:
        raise ValueError("current_step_id is required")

    current_step = plan[current_step_index]
    if current_step["step_id"] != current_step_id:
        raise ValueError("current step cursor does not match plan")

    return current_step_index, current_step


def _reason_about_next_turn(
    state: AgentState,
    question: str,
    step_id: str,
    task: str,
    current_react_trace: list[dict[str, Any]],
) -> dict[str, Any]:
    payload = {
        "question": question,
        "current_step_id": step_id,
        "task": task,
        "current_react_trace": current_react_trace,
        "current_correction_instruction": state.get("current_correction_instruction"),
    }
    data = _chat_json(
        system_prompt=REACT_REASONING_PROMPT,
        payload=payload,
    )
    thought = _required_value(data.get("thought"), "thought", str, allow_empty=False)
    need = _required_value(data.get("need"), "need", str)
    done = _required_value(data.get("done"), "reasoning.done", bool)

    return {
        "thought": thought,
        "need": need,
        "done": done,
    }


def _choose_action(
    question: str,
    step_id: str,
    task: str,
    current_react_trace: list[dict[str, Any]],
    reasoning: dict[str, Any],
) -> dict[str, Any]:
    payload = {
        "question": question,
        "current_step_id": step_id,
        "task": task,
        "current_react_trace": current_react_trace,
        "reasoning": reasoning,
        "available_tools": _available_tools(),
    }
    data = _chat_json(
        system_prompt=REACT_ACTING_PROMPT,
        payload=payload,
    )
    action = _required_value(data.get("action"), "action", str, allow_empty=False)
    action_input = _required_value(
        data.get("action_input", {}),
        "acting.action_input",
        dict,
    )
    if action != "none" and action not in TOOL_REGISTRY:
        raise ValueError(f"unknown action: {action}")

    return {
        "action": action,
        "action_input": action_input,
    }


def _execute_action(
    action: str,
    action_input: dict[str, Any],
    document_id: str | None,
) -> str:
    if action == "none":
        return "No tool action executed."

    arguments = dict(action_input)
    if action == "retrieve_uploaded_document" and "document_id" not in arguments:
        arguments["document_id"] = document_id

    result = TOOL_REGISTRY[action](**arguments)
    return json.dumps(
        {
            "action": action,
            "result": result,
        },
        ensure_ascii=False,
        default=str,
    )


def _tool_calls_from_react_results(
    react_results: list[dict[str, Any]],
) -> list[str]:
    tool_calls: list[str] = []
    for react_result in react_results:
        action = react_result.get("action")
        if not isinstance(action, str) or action == "none":
            continue
        if action not in tool_calls:
            tool_calls.append(action)
    return tool_calls


def _summarize_turn_result(
    question: str,
    step_id: str,
    task: str,
    current_react_trace: list[dict[str, Any]],
    reasoning: dict[str, Any],
    acting: dict[str, Any],
    observation: str,
) -> dict[str, Any]:
    payload = {
        "question": question,
        "current_step_id": step_id,
        "task": task,
        "reasoning": reasoning,
        "acting": acting,
        "observation": observation,
        "current_react_trace": current_react_trace,
    }
    data = _chat_json(
        system_prompt=REACT_RESULT_SUMMARY_PROMPT,
        payload=payload,
    )
    result = _required_value(data.get("result"), "result", str, allow_empty=False)
    success = _required_value(data.get("success"), "summary.success", bool)

    return {
        "result": result,
        "success": success,
    }


def _chat_json(system_prompt: str, payload: dict[str, Any]) -> dict[str, Any]:
    content = _chat_completion(
        system_prompt=system_prompt,
        user_message=json.dumps(payload, ensure_ascii=False),
        response_format={"type": "json_object"},
    )
    try:
        data = json.loads(content)
    except json.JSONDecodeError as exc:
        raise ValueError("react returned invalid JSON") from exc
    if not isinstance(data, dict):
        raise ValueError("react returned non-object JSON")
    return data


def _update_step_result(
    plan: list[PlanStepState],
    current_step_index: int,
    result: str,
) -> list[PlanStepState]:
    updated_plan = list(plan)
    step = updated_plan[current_step_index]
    try:
        updated_plan[current_step_index] = PlanStep(
            step_id=step["step_id"],
            task=step["task"],
            status="done",
            result=result,
            retry_count=step.get("retry_count", 0),
        ).model_dump()
    except (TypeError, ValidationError) as exc:
        raise ValueError("current plan step is invalid") from exc

    return updated_plan


def _required_value(
    value: Any,
    field: str,
    expected_type: type[Any],
    *,
    allow_empty: bool = True,
) -> Any:
    if not isinstance(value, expected_type):
        raise ValueError(f"{field} type mismatch")
    if isinstance(value, str) and not allow_empty and not value.strip():
        raise ValueError(f"{field} cannot be empty")
    return value


def _react_failed(state: AgentState, error: str) -> AgentState:
    message = f"react failed: {error}"
    failure = AgentFailure(
        reason="react_failed",
        message=message,
        node="react_node",
        target_step_id=state.get("current_step_id"),
    )
    return {
        "phase": "failed",
        "agent_status": "failed",
        "error": message,
        "failure": failure.model_dump(),
        "logs": add_log(
            state=state,
            node="react_node",
            message="react failed",
            extra={"error": error},
        ),
    }
"""