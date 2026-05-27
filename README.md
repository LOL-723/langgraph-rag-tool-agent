流程图
START
  ↓
router_node
  ↓
├── rag_node → answer_node
├── tool_selector_node → tool_executor_node → answer_node
└── answer_node
        ↓
  verifier_node
        ↓
├── finish → END
├── retry_answer → answer_node
├── retry_retrieve → rag_node
├── retry_tool → tool_selector_node
├── retry_router → router_node
└── fail → END
