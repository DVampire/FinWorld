workdir = "workdir"
tag = "general"
exp_path = f"{workdir}/{tag}"
log_path = "finworld.log"

tools = [
    dict(
        type="python_interpreter_tool",
    ),
    dict(
        type="web_fetcher_tool",
    )
]

web_fetcher_tool_config = dict(
    type="web_fetcher_tool",
)

agent = dict(
    type="GeneralAgent",
    model="gpt-4.1",
    name="general_agent",
    description="This is a general agent that can use various tools to solve tasks.",
    tools=tools,
    max_steps=20,
    provide_run_summary=True,
    template_path="finworld/agent/llm_agent/general_agent/prompts/general_agent.yaml",
)