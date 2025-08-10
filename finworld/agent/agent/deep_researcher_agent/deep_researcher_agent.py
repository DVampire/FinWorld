from typing import (
    Any,
)
import yaml

from finworld.agent.agent.general_agent import GeneralAgent
from finworld.base.async_multistep_agent import PromptTemplates

from finworld.memory import AgentMemory
from finworld.models import Model
from finworld.registry import AGENT
from finworld.utils import assemble_project_path

@AGENT.register_module(force=True)
class DeepResearcherAgent(GeneralAgent):
    def __init__(
            self,
            template_path: str,
            tools: list[Any],
            model: Model,
            prompt_templates: PromptTemplates | None = None,
            planning_interval: int | None = None,
            stream_outputs: bool = False,
            max_tool_threads: int | None = None,
            **kwargs,
    ):
        self.template_path = template_path

        super(DeepResearcherAgent, self).__init__(
            template_path = template_path,
            tools=tools,
            model=model,
            prompt_templates=prompt_templates,
            planning_interval=planning_interval,
            **kwargs,
        )

        template_path = assemble_project_path(self.template_path)
        with open(template_path, "r") as f:
            self.prompt_templates = yaml.safe_load(f)

        self.system_prompt = self.initialize_system_prompt()
        self.user_prompt = self.initialize_user_prompt()

        # Streaming setup
        self.stream_outputs = stream_outputs
        if self.stream_outputs and not hasattr(self.model, "generate_stream"):
            raise ValueError(
                "`stream_outputs` is set to True, but the model class implements no `generate_stream` method."
            )
        # Tool calling setup
        self.max_tool_threads = max_tool_threads

        self.memory = AgentMemory(
            system_prompt=self.system_prompt,
            user_prompt=self.user_prompt,
        )