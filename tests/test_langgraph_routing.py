import unittest
from unittest.mock import patch

from llm import langgraph


class LangGraphRoutingTest(unittest.TestCase):
    def test_use_rag_routes_to_agent_without_tool_calls(self) -> None:
        update = langgraph.router_node(
            {
                "question": "summarize this document",
                "use_rag": True,
                "file_info": {"document_id": "doc_123"},
                "logs": [],
            }
        )

        self.assertEqual(update["route"], "agent")
        self.assertEqual(update["tool_calls"], [])
        self.assertEqual(langgraph.route_decision(update), "agent_node")

    def test_tool_route_does_not_expose_rag_tool(self) -> None:
        tool_names = [tool["name"] for tool in langgraph._available_tools()]

        self.assertEqual(
            tool_names,
            ["get_current_time", "calculate_expression", "get_today_weather"],
        )
        self.assertNotIn("retrieve_uploaded_document", tool_names)

    def test_tool_executor_ignores_rag_tool_call(self) -> None:
        update = langgraph.tool_executor_node(
            {
                "tool_calls": [
                    {
                        "name": "retrieve_uploaded_document",
                        "arguments": {"query": "secret"},
                    }
                ],
                "logs": [],
            }
        )

        self.assertEqual(update["tool_results"], [])

    def test_agent_node_returns_final_result_without_verifier(self) -> None:
        def fake_planner(state):
            return {
                "plan": [
                    {
                        "step_id": "step_1",
                        "task": "answer",
                        "status": "pending",
                        "result": None,
                        "retry_count": 0,
                    }
                ],
                "plan_revision": 1,
                "plan_updates": [],
                "planner_mode": "initial",
                "agent_status": "running",
                "logs": [],
            }

        def fake_select(state):
            if state.get("step_results"):
                return {
                    "should_continue_next": "finish",
                    "current_step_index": 1,
                    "current_step_id": None,
                    "agent_status": "running",
                    "logs": [],
                }
            return {
                "should_continue_next": "continue",
                "current_step_index": 0,
                "current_step_id": "step_1",
                "agent_status": "running",
                "logs": [],
            }

        def fake_loop(state):
            plan = list(state["plan"])
            plan[0] = dict(plan[0], status="done", result="step answer")
            return {
                "plan": plan,
                "step_results": [
                    {
                        "step_id": "step_1",
                        "task": "answer",
                        "result": "step answer",
                    }
                ],
                "agent_status": "running",
                "logs": [],
            }

        with (
            patch("llm.langgraph.planner_node", side_effect=fake_planner),
            patch("llm.langgraph.select_next_step_node", side_effect=fake_select),
            patch("llm.langgraph.agent_loop_node", side_effect=fake_loop),
            patch("llm.langgraph._summarize_agent_answer", return_value="final answer"),
        ):
            update = langgraph.agent_node(
                {
                    "question": "complex task",
                    "use_rag": True,
                    "file_info": {"document_id": "doc_123"},
                    "logs": [],
                }
            )

        self.assertEqual(update["route"], "agent")
        self.assertEqual(update["answer"], "final answer")
        self.assertEqual(update["end_status"], "finished")
        self.assertEqual(update["agent_state"]["document_id"], "doc_123")


if __name__ == "__main__":
    unittest.main()
