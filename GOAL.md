planner_node
  ↓
select_next_step_node
  ↓
react_node
  ↓
step_reflection_node
  ↓
route_step_reflection
      ├── pass → plan_update_node
      ├── retry_react → react_node
      ├── replan → planner_node / plan_update_node
      └── fail → agent_fail_node
  ↓
should_continue
      ├── continue → select_next_step_node
      └── finish → final_answer_node
  ↓
final_reflection_node
  ↓
route_final_reflection
      ├── pass → agent_gate_node
      ├── rewrite_final → rewrite_final_node → agent_gate_node
      ├── retry_react → select_next_step_node / react_node
      ├── replan → planner_node
      └── fail → agent_fail_node