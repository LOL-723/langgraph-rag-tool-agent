import json
from typing import Any

from pydantic import ValidationError

from llm.Agent.nodes.universal import _available_tools, _chat_completion, add_log
from llm.Agent.prompt import AGENT_LOOP_PROMPT
from llm.Agent.state import (
    AgentFailure,
    AgentState,
    MAX_REACT_TURNS_PER_STEP,
    MAX_REPLAN_COUNT,
    MAX_STEP_REPLAN_COUNT,
    AgentLoopResult,
    AgentLoopSignal,
    PlanStep,
    PlanStepState,
)
from llm.tools import TOOL_REGISTRY


def agent_loop_node(state: AgentState) -> AgentState:
    try:
        current_step_index, current_step = _current_step(state)
        question = state.get("question", "")
        step_id = current_step["step_id"]
        task = current_step["task"]
        react_results = list(state.get("react_results", []))
        step_results = list(state.get("step_results", []))
        overthink_counts = dict(state.get("overthink_counts", {}))
        no_finding_counts = dict(state.get("no_finding_counts", {}))
        failed_tools = list(state.get("failed_tools", []))
        subagent_results = list(state.get("subagent_results", []))
        agent_depth = int(state.get("agent_depth", 0) or 0)
        consecutive_think_count = 0
        turn_index = 0

        while turn_index < MAX_REACT_TURNS_PER_STEP:
            decision = _decide_next_loop(
                state=state,
                question=question,
                step_id=step_id,
                task=task,
                react_results=react_results,
                overthink_count=overthink_counts.get(step_id, 0),
                no_finding_count=no_finding_counts.get(step_id, 0),
            )
            turn_index += 1
            loop_result = AgentLoopResult(
                thought=decision["thought"],
                decide_type=decision["decide_type"],
                Signal=decision["Signal"],
                no_finding=decision["no_finding"],
                tool_name=decision["tool_name"],
                arguments=decision["arguments"],
                answer=decision["answer"],
            ).model_dump()
            _print_agent_thought_trace(
                step_id=step_id,
                turn_index=turn_index,
                decision=decision,
            )

            signal = decision["Signal"]
            if signal == "overturning":
                return _handle_overturning_signal(
                    state=state,
                    step_id=step_id,
                    task=task,
                    turn_index=turn_index,
                    loop_result=loop_result,
                    react_results=react_results,
                    failed_tools=failed_tools,
                    overthink_counts=overthink_counts,
                    no_finding_counts=no_finding_counts,
                    subagent_results=subagent_results,
                )
            if signal == "overthink":
                signal_result = _handle_signal(
                    state=state,
                    step_id=step_id,
                    overthink_counts=overthink_counts,
                )
                if signal_result is not None:
                    return signal_result
                react_results = []
                consecutive_think_count = 0
                turn_index = 0
                continue
            if signal == "tool_error":
                raise ValueError("Signal tool_error must be triggered by a real tool observation error")
            if signal is not None:
                raise ValueError(f"Signal is not implemented yet: {signal}")

            no_finding_signal = _update_no_finding_counts(
                step_id=step_id,
                no_finding=decision["no_finding"],
                no_finding_counts=no_finding_counts,
            )
            if no_finding_signal is not None:
                return _handle_finding_missing_signal(
                    state=state,
                    step_id=step_id,
                    task=task,
                    turn_index=turn_index,
                    loop_result=loop_result,
                    react_results=react_results,
                    failed_tools=failed_tools,
                    overthink_counts=overthink_counts,
                    no_finding_counts=no_finding_counts,
                    subagent_results=subagent_results,
                )

            decide_type = decision["decide_type"]
            if decide_type == "think":
                consecutive_think_count += 1
                if consecutive_think_count >= 3:
                    signal_result = _handle_signal(
                        signal="overthink",
                        state=state,
                        step_id=step_id,
                        overthink_counts=overthink_counts,
                    )
                    if signal_result is not None:
                        return signal_result
                    react_results = []
                    consecutive_think_count = 0
                    turn_index = 0
                    continue
                react_results.append(loop_result)
                continue

            if decide_type == "tool_call":
                consecutive_think_count = 0
                observation = _execute_tool_call(
                    tool_name=decision["tool_name"],
                    arguments=decision["arguments"],
                    document_id=state.get("document_id"),
                )
                loop_result["observation"] = observation
                react_results.append(loop_result)
                tool_error = _observation_error(observation)
                if tool_error is not None:
                    loop_result["Signal"] = "tool_error"
                    tool_name = _required_value(
                        decision["tool_name"],
                        "tool_name",
                        str,
                        allow_empty=False,
                    )
                    failed_tools = _append_unique(failed_tools, tool_name)
                    if agent_depth >= 1:
                        failed_state = dict(state)
                        failed_state["failed_tools"] = failed_tools
                        return _agent_loop_failed(
                            failed_state,
                            f"tool_error cannot create subagent at agent_depth {agent_depth}",
                        )
                    subagent_update = _run_tool_error_subagent(
                        parent_state=state,
                        question=question,
                        step_id=step_id,
                        task=task,
                        react_results=react_results,
                        step_results=step_results,
                        failed_tools=failed_tools,
                        overthink_counts=overthink_counts,
                    )
                    if subagent_update.get("agent_status") == "failed":
                        failed_state = dict(state)
                        failed_state["failed_tools"] = failed_tools
                        failed_state["subagent_results"] = subagent_results + [
                            {
                                "step_id": step_id,
                                "task": task,
                                "status": "failed",
                                "error": subagent_update.get("error"),
                            }
                        ]
                        return _agent_loop_failed(
                            failed_state,
                            "tool_error subagent failed: "
                            f"{subagent_update.get('error') or 'unknown error'}",
                        )
                    subagent_answer = _subagent_answer(subagent_update, step_id)
                    subagent_results.append(
                        {
                            "step_id": step_id,
                            "task": task,
                            "status": "done",
                            "result": subagent_answer,
                        }
                    )
                    updated_plan = _update_step_result(
                        plan=list(state.get("plan", [])),
                        current_step_index=current_step_index,
                        result=subagent_answer,
                    )
                    step_results.append(
                        {
                            "step_id": step_id,
                            "task": task,
                            "result": subagent_answer,
                        }
                    )
                    return {
                        "plan": updated_plan,
                        "current_react_turn_count": turn_index,
                        "react_results": [],
                        "tool_calls": _tool_calls_from_loop_results(react_results),
                        "step_results": step_results,
                        "overthink_counts": overthink_counts,
                        "no_finding_counts": no_finding_counts,
                        "failed_tools": failed_tools,
                        "subagent_results": subagent_results,
                        "phase": "reacting",
                        "agent_status": "running",
                        "logs": add_log(
                            state=state,
                            node="agent_loop_node",
                            message="step completed by tool_error subagent",
                            extra={
                                "current_step_id": step_id,
                                "failed_tools": failed_tools,
                            },
                        ),
                    }
                continue

            if decide_type == "finish":
                consecutive_think_count = 0
                answer = _required_value(
                    decision["answer"],
                    "answer",
                    str,
                    allow_empty=False,
                )
                updated_plan = _update_step_result(
                    plan=list(state.get("plan", [])),
                    current_step_index=current_step_index,
                    result=answer,
                )
                step_results.append(
                    {
                        "step_id": step_id,
                        "task": task,
                        "result": answer,
                    }
                )
                return {
                    "plan": updated_plan,
                    "current_react_turn_count": turn_index,
                    "react_results": [],
                    "tool_calls": _tool_calls_from_loop_results(react_results),
                    "step_results": step_results,
                    "overthink_counts": overthink_counts,
                    "no_finding_counts": no_finding_counts,
                    "failed_tools": failed_tools,
                    "subagent_results": subagent_results,
                    "phase": "reacting",
                    "agent_status": "running",
                    "logs": add_log(
                        state=state,
                        node="agent_loop_node",
                        message="step agent loop completed",
                        extra={
                            "current_step_id": step_id,
                            "agent_loop_turn_count": turn_index,
                        },
                    ),
                }

            if decide_type == "fail":
                raise ValueError("agent loop decision returned fail")

        raise ValueError(
            f"agent loop exceeded {MAX_REACT_TURNS_PER_STEP} turns before step completed"
        )
    except Exception as exc:
        return _agent_loop_failed(state, str(exc))


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


def _decide_next_loop(
    state: AgentState,
    question: str,
    step_id: str,
    task: str,
    react_results: list[dict[str, Any]],
    overthink_count: int,
    no_finding_count: int,
) -> dict[str, Any]:
    payload = {
        "question": question,
        "current_step_id": step_id,
        "task": task,
        "completed_steps": state.get("step_results", []),
        "react_results": react_results,
        "previous_thought": _previous_thought(react_results),
        "no_finding_count": no_finding_count,
        "current_correction_instruction": state.get("current_correction_instruction"),
        "overthink_count": overthink_count,
        "failed_tools": state.get("failed_tools", []),
        "agent_depth": state.get("agent_depth", 0),
        "available_tools": _available_tools(state.get("failed_tools", [])),
    }
    data = _chat_json(
        system_prompt=AGENT_LOOP_PROMPT,
        payload=payload,
    )
    thought = _required_value(data.get("thought"), "thought", str, allow_empty=False)
    decide_type = _required_value(
        data.get("decide_type"),
        "decide_type",
        str,
        allow_empty=False,
    )
    if decide_type not in {"think", "tool_call", "finish", "fail"}:
        raise ValueError(f"unknown decide_type: {decide_type}")
    signal = _optional_signal(data.get("Signal"))
    no_finding = _optional_binary_int(data.get("no_finding", 0), "no_finding")

    tool_name = data.get("tool_name")
    if decide_type == "tool_call":
        tool_name = _required_value(
            tool_name,
            "tool_name",
            str,
            allow_empty=False,
        )
        if tool_name not in TOOL_REGISTRY or tool_name in state.get("failed_tools", []):
            raise ValueError(f"unknown tool_name: {tool_name}")
    elif tool_name is not None:
        raise ValueError("tool_name must be null unless decide_type is tool_call")

    arguments = _required_value(
        data.get("arguments", {}),
        "arguments",
        dict,
    )
    answer = _required_value(
        data.get("answer", ""),
        "answer",
        str,
    )

    return {
        "thought": thought,
        "decide_type": decide_type,
        "Signal": signal,
        "no_finding": no_finding,
        "tool_name": tool_name,
        "arguments": arguments,
        "answer": answer,
    }


def _previous_thought(react_results: list[dict[str, Any]]) -> str | None:
    for react_result in reversed(react_results):
        thought = react_result.get("thought")
        if isinstance(thought, str) and thought.strip():
            return thought
    return None


def _print_agent_thought_trace(
    step_id: str,
    turn_index: int,
    decision: dict[str, Any],
) -> None:
    signal = decision.get("Signal") or "none"
    tool_name = decision.get("tool_name") or "none"
    print(
        "[Agent Thought] "
        f"step={step_id} "
        f"turn={turn_index} "
        f"decision={decision.get('decide_type')} "
        f"signal={signal} "
        f"tool={tool_name}\n"
        f"{decision.get('thought', '')}",
        flush=True,
    )


def _optional_binary_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field} type mismatch")
    if value not in {0, 1}:
        raise ValueError(f"{field} must be 0 or 1")
    return value


def _optional_signal(value: Any) -> AgentLoopSignal | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError("Signal type mismatch")
    if value not in {"overthink", "tool_error", "overturning", "finding_missing"}:
        raise ValueError(f"unknown Signal: {value}")
    return value


def _handle_signal(
    state: AgentState,
    step_id: str,
    overthink_counts: dict[str, int],
) -> AgentState | None:
    next_count = overthink_counts.get(step_id, 0) + 1
    overthink_counts[step_id] = next_count
    if next_count > 1:
        failed_state = dict(state)
        failed_state["overthink_counts"] = overthink_counts
        return _agent_loop_failed(
            failed_state,
            "agent loop triggered overthink more than once in the same plan step",
        )
    return None


def _update_no_finding_counts(
    step_id: str,
    no_finding: int,
    no_finding_counts: dict[str, int],
) -> AgentLoopSignal | None:
    if no_finding == 0:
        no_finding_counts[step_id] = 0
        return None

    next_count = no_finding_counts.get(step_id, 0) + 1
    no_finding_counts[step_id] = next_count
    if next_count >= 6:
        return "finding_missing"
    return None


def _handle_overturning_signal(
    state: AgentState,
    step_id: str,
    task: str,
    turn_index: int,
    loop_result: dict[str, Any],
    react_results: list[dict[str, Any]],
    failed_tools: list[str],
    overthink_counts: dict[str, int],
    no_finding_counts: dict[str, int],
    subagent_results: list[dict[str, Any]],
) -> AgentState:
    replan_count = int(state.get("replan_count", 0) or 0)
    if replan_count >= MAX_REPLAN_COUNT:
        failed_state = dict(state)
        failed_state["replan_count"] = replan_count
        failed_state["no_finding_counts"] = no_finding_counts
        return _agent_loop_failed(
            failed_state,
            "agent loop triggered overturning replan more than once",
        )

    trigger_trace = react_results + [loop_result]
    last_observation = _last_tool_observation(react_results)
    return {
        "planner_mode": "replan",
        "replan_count": replan_count + 1,
        "replan_context": {
            "signal": "overturning",
            "current_step_id": step_id,
            "current_step": {"step_id": step_id, "task": task},
            "react_results": trigger_trace,
            "last_tool_observation": last_observation,
        },
        "last_tool_observation": last_observation,
        "current_react_turn_count": turn_index,
        "react_results": [],
        "tool_calls": _tool_calls_from_loop_results(react_results),
        "overthink_counts": overthink_counts,
        "no_finding_counts": no_finding_counts,
        "failed_tools": failed_tools,
        "subagent_results": subagent_results,
        "phase": "replanning",
        "agent_status": "running",
        "logs": add_log(
            state=state,
            node="agent_loop_node",
            message="overturning signal requested full replan",
            extra={
                "current_step_id": step_id,
                "replan_count": replan_count + 1,
            },
        ),
    }


def _handle_finding_missing_signal(
    state: AgentState,
    step_id: str,
    task: str,
    turn_index: int,
    loop_result: dict[str, Any],
    react_results: list[dict[str, Any]],
    failed_tools: list[str],
    overthink_counts: dict[str, int],
    no_finding_counts: dict[str, int],
    subagent_results: list[dict[str, Any]],
) -> AgentState:
    step_replan_count = int(state.get("step_replan_count", 0) or 0)
    if step_replan_count >= MAX_STEP_REPLAN_COUNT:
        failed_state = dict(state)
        failed_state["step_replan_count"] = step_replan_count
        failed_state["no_finding_counts"] = no_finding_counts
        return _agent_loop_failed(
            failed_state,
            "agent loop triggered finding_missing step replan more than once",
        )

    trigger_trace = react_results + [loop_result]
    return {
        "planner_mode": "step_replan",
        "step_replan_count": step_replan_count + 1,
        "replan_context": {
            "signal": "finding_missing",
            "current_step_id": step_id,
            "current_step": {"step_id": step_id, "task": task},
            "react_results": trigger_trace,
            "no_finding_count": no_finding_counts.get(step_id, 0),
        },
        "current_react_turn_count": turn_index,
        "react_results": trigger_trace,
        "tool_calls": _tool_calls_from_loop_results(react_results),
        "overthink_counts": overthink_counts,
        "no_finding_counts": no_finding_counts,
        "failed_tools": failed_tools,
        "subagent_results": subagent_results,
        "phase": "replanning",
        "agent_status": "running",
        "logs": add_log(
            state=state,
            node="agent_loop_node",
            message="finding_missing signal requested step replan",
            extra={
                "current_step_id": step_id,
                "step_replan_count": step_replan_count + 1,
                "no_finding_count": no_finding_counts.get(step_id, 0),
            },
        ),
    }


def _last_tool_observation(react_results: list[dict[str, Any]]) -> str | None:
    for react_result in reversed(react_results):
        observation = react_result.get("observation")
        if isinstance(observation, str) and observation.strip():
            return observation
    return None


def _observation_error(observation: str) -> str | None:
    try:
        data = json.loads(observation)
    except json.JSONDecodeError as exc:
        raise ValueError("tool observation returned invalid JSON") from exc
    if not isinstance(data, dict):
        raise ValueError("tool observation returned non-object JSON")
    error = data.get("error")
    if error is None:
        return None
    if not isinstance(error, str):
        raise ValueError("tool observation error type mismatch")
    if not error.strip():
        return None
    return error


def _append_unique(values: list[str], value: str) -> list[str]:
    if value in values:
        return values
    return values + [value]


def _run_tool_error_subagent(
    parent_state: AgentState,
    question: str,
    step_id: str,
    task: str,
    react_results: list[dict[str, Any]],
    step_results: list[dict[str, Any]],
    failed_tools: list[str],
    overthink_counts: dict[str, int],
) -> AgentState:
    subagent_step = PlanStep(step_id=step_id, task=task).model_dump()
    subagent_state: AgentState = {
        "question": question,
        "document_id": parent_state.get("document_id"),
        "plan": [subagent_step],
        "current_step_index": 0,
        "current_step_id": step_id,
        "react_results": list(react_results),
        "step_results": list(step_results),
        "failed_tools": list(failed_tools),
        "overthink_counts": dict(overthink_counts),
        "no_finding_counts": dict(parent_state.get("no_finding_counts", {})),
        "agent_depth": 1,
        "logs": parent_state.get("logs", []),
    }
    return agent_loop_node(subagent_state)


def _subagent_answer(subagent_update: AgentState, step_id: str) -> str:
    for step_result in reversed(subagent_update.get("step_results", [])):
        if step_result.get("step_id") != step_id:
            continue
        result = step_result.get("result")
        if isinstance(result, str) and result.strip():
            return result
    raise ValueError("tool_error subagent returned no usable result")


def _execute_tool_call(
    tool_name: str | None,
    arguments: dict[str, Any],
    document_id: str | None,
) -> str:
    if tool_name is None:
        raise ValueError("tool_name is required for tool_call")

    tool_arguments = dict(arguments)
    if tool_name == "retrieve_uploaded_document" and "document_id" not in tool_arguments:
        tool_arguments["document_id"] = document_id

    try:
        result = TOOL_REGISTRY[tool_name](**tool_arguments)
    except Exception as exc:
        return json.dumps(
            {
                "tool_name": tool_name,
                "result": None,
                "error": str(exc),
            },
            ensure_ascii=False,
            default=str,
        )

    return json.dumps(
        {
            "tool_name": tool_name,
            "result": result,
            "error": None,
        },
        ensure_ascii=False,
        default=str,
    )


def _tool_calls_from_loop_results(
    react_results: list[dict[str, Any]],
) -> list[str]:
    tool_calls: list[str] = []
    for react_result in react_results:
        tool_name = react_result.get("tool_name")
        if not isinstance(tool_name, str):
            continue
        if tool_name not in tool_calls:
            tool_calls.append(tool_name)
    return tool_calls


def _chat_json(system_prompt: str, payload: dict[str, Any]) -> dict[str, Any]:
    content = _chat_completion(
        system_prompt=system_prompt,
        user_message=json.dumps(payload, ensure_ascii=False),
        response_format={"type": "json_object"},
    )
    try:
        data = json.loads(content)
    except json.JSONDecodeError as exc:
        raise ValueError("agent loop returned invalid JSON") from exc
    if not isinstance(data, dict):
        raise ValueError("agent loop returned non-object JSON")
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


def _agent_loop_failed(state: AgentState, error: str) -> AgentState:
    message = f"agent loop failed: {error}"
    failure = AgentFailure(
        reason="react_failed",
        message=message,
        node="agent_loop_node",
        target_step_id=state.get("current_step_id"),
    )
    failed_update: AgentState = {
        "phase": "failed",
        "agent_status": "failed",
        "error": message,
        "failure": failure.model_dump(),
        "logs": add_log(
            state=state,
            node="agent_loop_node",
            message="agent loop failed",
            extra={"error": error},
        ),
    }
    if "failed_tools" in state:
        failed_update["failed_tools"] = state["failed_tools"]
    if "subagent_results" in state:
        failed_update["subagent_results"] = state["subagent_results"]
    if "overthink_counts" in state:
        failed_update["overthink_counts"] = state["overthink_counts"]
    if "no_finding_counts" in state:
        failed_update["no_finding_counts"] = state["no_finding_counts"]
    if "replan_count" in state:
        failed_update["replan_count"] = state["replan_count"]
    if "step_replan_count" in state:
        failed_update["step_replan_count"] = state["step_replan_count"]
    return failed_update
