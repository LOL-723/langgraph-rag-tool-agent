Flow

START
  -> router_node
      -> agent_node -> END
      -> tool_selector_node -> tool_executor_node -> answer_node
      -> answer_node
  -> verifier_node
      -> finish -> END
      -> retry_answer -> answer_node
      -> retry_tool -> tool_selector_node
      -> retry_router -> router_node
      -> fail -> END

Uploaded-document retrieval is only available inside the Agent route through
the `retrieve_uploaded_document` tool. The normal tool route exposes only time,
calculator, and weather tools.
