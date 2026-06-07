import unittest
from typing import Any

from llm.Agent.prompt import (
    FINAL_RESULT_SUMMARY_PROMPT,
    PLANNER_PROMPT,
    REFLECTION_PROMPT,
    REACT_ACTING_PROMPT,
    REACT_REASONING_PROMPT,
    REACT_RESULT_SUMMARY_PROMPT,
)
from llm.Agent.state import (
    AgentPlan,
    AgentFailure,
    AgentState,
    FinalGateDecision,
    MAX_FINAL_REWRITE_COUNT,
    MAX_PLAN_STEPS,
    MAX_REACT_RETRY_COUNT,
    MAX_REACT_TURNS_PER_STEP,
    MAX_REPLAN_COUNT,
    PlanStep,
    PlanUpdate,
    ReactResult,
    ReflectionDecision,
)


class AgentStateTest(unittest.TestCase):
    def test_agent_loop_limits_are_defined(self) -> None:
        self.assertEqual(MAX_PLAN_STEPS, 8)
        self.assertEqual(MAX_REPLAN_COUNT, 2)
        self.assertEqual(MAX_REACT_TURNS_PER_STEP, 4)
        self.assertEqual(MAX_REACT_RETRY_COUNT, 2)
        self.assertEqual(MAX_FINAL_REWRITE_COUNT, 1)

    def test_planner_prompt_matches_plan_contract(self) -> None:
        self.assertIn(f"no more than {MAX_PLAN_STEPS} steps", PLANNER_PROMPT)
        self.assertIn('"steps"', PLANNER_PROMPT)
        self.assertIn('"step_id": "step_1"', PLANNER_PROMPT)
        self.assertIn('"task": "specific task"', PLANNER_PROMPT)
        self.assertIn("available tools", PLANNER_PROMPT)

    def test_react_reasoning_prompt_matches_thought_contract(self) -> None:
        self.assertIn(
            f"at most {MAX_REACT_TURNS_PER_STEP} ReAct turns",
            REACT_REASONING_PROMPT,
        )
        self.assertIn("current_react_trace", REACT_REASONING_PROMPT)
        self.assertIn('"thought"', REACT_REASONING_PROMPT)
        self.assertIn('"need"', REACT_REASONING_PROMPT)
        self.assertIn('"done"', REACT_REASONING_PROMPT)
        self.assertIn("Do not choose tools.", REACT_REASONING_PROMPT)
        self.assertIn("Do not output action or action_input.", REACT_REASONING_PROMPT)
        self.assertIn("Do not fabricate tool results or observations.", REACT_REASONING_PROMPT)

    def test_react_acting_prompt_matches_action_contract(self) -> None:
        self.assertIn("available_tools", REACT_ACTING_PROMPT)
        self.assertIn("reasoning.done is true", REACT_ACTING_PROMPT)
        self.assertIn('return action as "none"', REACT_ACTING_PROMPT)
        self.assertIn('The action must be one tool name from available_tools, or "none".', REACT_ACTING_PROMPT)
        self.assertIn("Do not invent tools that are not in available_tools.", REACT_ACTING_PROMPT)
        self.assertIn("Do not output thought, need, done, observation, result, success", REACT_ACTING_PROMPT)
        self.assertIn("Do not fabricate tool results or observations.", REACT_ACTING_PROMPT)
        self.assertIn('"action"', REACT_ACTING_PROMPT)
        self.assertIn('"action_input"', REACT_ACTING_PROMPT)

    def test_react_result_summary_prompt_matches_result_contract(self) -> None:
        self.assertIn("observation", REACT_RESULT_SUMMARY_PROMPT)
        self.assertIn("current_react_trace", REACT_RESULT_SUMMARY_PROMPT)
        self.assertIn("Do not fabricate tool results or observations.", REACT_RESULT_SUMMARY_PROMPT)
        self.assertIn("success does not mean the whole step passed final review.", REACT_RESULT_SUMMARY_PROMPT)
        self.assertIn("Do not decide pass, retry_react, replan, or fail.", REACT_RESULT_SUMMARY_PROMPT)
        self.assertIn("Do not output thought, need, action, action_input, observation", REACT_RESULT_SUMMARY_PROMPT)
        self.assertIn('"result"', REACT_RESULT_SUMMARY_PROMPT)
        self.assertIn('"success"', REACT_RESULT_SUMMARY_PROMPT)

    def test_reflection_prompt_matches_review_contract(self) -> None:
        self.assertIn("scope", REFLECTION_PROMPT)
        self.assertIn("expected_output_schema", REFLECTION_PROMPT)
        self.assertIn("output_to_check", REFLECTION_PROMPT)
        self.assertIn("format_valid", REFLECTION_PROMPT)
        self.assertIn("grounded", REFLECTION_PROMPT)
        self.assertIn("relevant", REFLECTION_PROMPT)
        self.assertIn('"problem"', REFLECTION_PROMPT)
        self.assertIn('"correction_instruction"', REFLECTION_PROMPT)
        self.assertIn("Do not modify plan, step, ReactResult, observation, result, or final answer.", REFLECTION_PROMPT)
        self.assertIn("Do not generate a replacement answer.", REFLECTION_PROMPT)
        self.assertIn("Do not fabricate evidence, tool results, observations, or step results.", REFLECTION_PROMPT)
        self.assertIn("The only business text you may create is problem.", REFLECTION_PROMPT)
        self.assertIn("correction_instruction must always be null.", REFLECTION_PROMPT)

    def test_final_result_summary_prompt_matches_answer_contract(self) -> None:
        self.assertIn("question", FINAL_RESULT_SUMMARY_PROMPT)
        self.assertIn("plan", FINAL_RESULT_SUMMARY_PROMPT)
        self.assertIn("step_results", FINAL_RESULT_SUMMARY_PROMPT)
        self.assertIn("react_results", FINAL_RESULT_SUMMARY_PROMPT)
        self.assertIn('"final_answer"', FINAL_RESULT_SUMMARY_PROMPT)
        self.assertIn("Use only step_results and real observations", FINAL_RESULT_SUMMARY_PROMPT)
        self.assertIn("Do not add facts that are not supported by step_results", FINAL_RESULT_SUMMARY_PROMPT)
        self.assertIn("Do not expose Thought, Action, Observation, Reflection", FINAL_RESULT_SUMMARY_PROMPT)
        self.assertIn("REFLECTION_PROMPT with scope", FINAL_RESULT_SUMMARY_PROMPT)
        self.assertIn("Do not modify plan, step_results, react_results, or observations.", FINAL_RESULT_SUMMARY_PROMPT)

    def test_agent_plan_dumps_serializable_steps(self) -> None:
        plan = AgentPlan(
            steps=[
                PlanStep(step_id="step_1", task="Search uploaded documents"),
                PlanStep(step_id="step_2", task="Summarize evidence"),
            ]
        )

        state: AgentState = {
            "question": "What does the document say?",
            "document_id": "doc_1",
            "plan": [step.model_dump() for step in plan.steps],
            "current_step_index": 0,
            "current_step_id": "step_1",
            "phase": "planning",
            "agent_status": "running",
        }

        self.assertEqual(state["plan"][0]["step_id"], "step_1")
        self.assertEqual(state["plan"][0]["status"], "pending")
        self.assertIsNone(state["plan"][0]["result"])

    def test_plan_revision_and_update_history_are_serializable(self) -> None:
        plan = AgentPlan(
            steps=[PlanStep(step_id="step_1", task="Read document")],
            revision=2,
            reason="Replanned after final reflection found missing evidence.",
        )
        plan_update = PlanUpdate(
            revision=2,
            reason="Added missing evidence step.",
            changed_steps=["step_1"],
        )

        state: AgentState = {
            "plan": [step.model_dump() for step in plan.steps],
            "plan_revision": plan.revision,
            "plan_updates": [plan_update.model_dump()],
            "phase": "replanning",
            "agent_status": "running",
        }

        self.assertEqual(state["plan_revision"], 2)
        self.assertEqual(state["plan_updates"][0]["changed_steps"], ["step_1"])

    def test_react_trace_supports_multiple_actions_per_step(self) -> None:
        first_result = ReactResult(
            step_id="step_1",
            thought="Need uploaded document context",
            need="Find evidence about the refund policy.",
            action="search_uploaded_docs",
            action_input={"query": "refund policy"},
            observation="Found one relevant paragraph",
            result="Partial evidence",
            success=True,
        )
        second_result = ReactResult(
            step_id="step_1",
            thought="Need more precise evidence",
            need="Find evidence about the refund deadline.",
            action="search_uploaded_docs",
            action_input={"query": "refund deadline"},
            observation="Found deadline details",
            result="Complete evidence",
            success=True,
        )

        current_react_trace: list[dict[str, Any]] = []
        current_react_trace = current_react_trace + [first_result.model_dump()]
        current_react_trace = current_react_trace + [second_result.model_dump()]

        state: AgentState = {
            "current_step_id": "step_1",
            "current_react_trace": current_react_trace,
            "phase": "reacting",
            "agent_status": "running",
        }

        self.assertEqual(len(state["current_react_trace"]), 2)
        self.assertEqual(state["current_react_trace"][0]["need"], "Find evidence about the refund policy.")
        self.assertEqual(state["current_react_trace"][1]["result"], "Complete evidence")

    def test_step_retry_state_tracks_route_and_correction(self) -> None:
        state: AgentState = {
            "current_step_id": "step_1",
            "step_reflection_next": "retry_react",
            "current_correction_instruction": "Search narrower terms.",
            "current_react_turn_count": 2,
            "react_retry_count": 1,
            "step_retry_counts": {"step_1": 1},
            "phase": "step_reflecting",
            "agent_status": "running",
        }

        self.assertEqual(state["step_reflection_next"], "retry_react")
        self.assertEqual(state["step_retry_counts"]["step_1"], 1)
        self.assertEqual(state["current_correction_instruction"], "Search narrower terms.")

    def test_reflection_decision_records_scope_and_route_intent(self) -> None:
        retry_decision = ReflectionDecision(
            scope="step",
            status="retry_react",
            severity="medium",
            target_step_id="step_1",
            problem="Evidence is incomplete",
            correction_instruction="Search for the deadline before answering.",
        )
        replan_decision = ReflectionDecision(
            scope="plan",
            status="replan",
            severity="high",
            problem="The original plan misses a comparison step.",
        )
        final_decision = ReflectionDecision(
            scope="final",
            status="rewrite_final",
            severity="medium",
            problem="The final answer omits step 2.",
        )

        state: AgentState = {
            "reflection_decisions": [
                retry_decision.model_dump(),
                replan_decision.model_dump(),
                final_decision.model_dump(),
            ],
            "replan_count": 1,
            "react_retry_count": 1,
            "final_rewrite_count": 1,
            "phase": "final_reflecting",
            "agent_status": "running",
        }

        self.assertEqual(state["reflection_decisions"][0]["status"], "retry_react")
        self.assertEqual(state["reflection_decisions"][1]["scope"], "plan")
        self.assertEqual(state["reflection_decisions"][2]["status"], "rewrite_final")
        self.assertEqual(state["replan_count"], 1)
        self.assertEqual(state["final_rewrite_count"], 1)

    def test_final_routes_gate_and_failure_are_serializable(self) -> None:
        gate_decision = FinalGateDecision(
            status="fail",
            problem="Final answer is not grounded in completed step results.",
            correction_instruction="Rewrite using step result evidence only.",
        )
        failure = AgentFailure(
            reason="final_gate_failed",
            message="Final gate rejected the answer after rewrite budget was exhausted.",
            node="agent_gate_node",
        )

        state: AgentState = {
            "should_continue_next": "finish",
            "final_reflection_next": "rewrite_final",
            "draft_final_answer": "Draft answer",
            "final_answer": "Final answer",
            "final_gate_decision": gate_decision.model_dump(),
            "failure": failure.model_dump(),
            "phase": "agent_gating",
            "agent_status": "failed",
        }

        self.assertEqual(state["should_continue_next"], "finish")
        self.assertEqual(state["final_reflection_next"], "rewrite_final")
        self.assertEqual(state["final_gate_decision"]["status"], "fail")
        self.assertEqual(state["failure"]["reason"], "final_gate_failed")


if __name__ == "__main__":
    unittest.main()
