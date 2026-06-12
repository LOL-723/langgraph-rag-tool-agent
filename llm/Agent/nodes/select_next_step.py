from llm.Agent.nodes.universal import add_log
from llm.Agent.state import AgentState, PlanStepState


def select_next_step_node(state: AgentState) -> AgentState:
    plan = state.get("plan", [])
    next_step = _find_next_pending_step(plan)

    if next_step is None:
        return {
            "current_step_index": len(plan),
            "current_step_id": None,
            "should_continue_next": "finish",
            "phase": "selecting_step",
            "agent_status": "running",
            "logs": add_log(
                state=state,
                node="select_next_step_node",
                message="no pending step found",
            ),
        }

    index, step = next_step
    return {
        "current_step_index": index,
        "current_step_id": step["step_id"],
        "should_continue_next": "continue",
        "phase": "selecting_step",
        "agent_status": "running",
        "logs": add_log(
            state=state,
            node="select_next_step_node",
            message="next pending step selected",
            extra={
                "current_step_index": index,
                "current_step_id": step["step_id"],
            },
        ),
    }


def _find_next_pending_step(
    plan: list[PlanStepState],
) -> tuple[int, PlanStepState] | None:
    for index, step in enumerate(plan):
        if step["status"] == "pending":
            return index, step
    return None
