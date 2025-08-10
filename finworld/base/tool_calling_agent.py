# coding=utf-8

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import importlib
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TYPE_CHECKING, Any
import yaml
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text
from collections.abc import Generator


if TYPE_CHECKING:
    import PIL.Image

from finworld.memory import (ActionStep,
                        ToolCall)
from finworld.models import (
    ChatMessage,
    ChatMessageStreamDelta,
)
from finworld.log import (
    LogLevel,
)

from finworld.tools import Tool
from finworld.exception import (
    AgentParsingError,
    AgentToolCallError,
    AgentToolExecutionError,
    AgentGenerationError,
    AgentExecutionError,
)
from finworld.base.multistep_agent import (MultiStepAgent,
                                      PromptTemplates,
                                      populate_template,
                                      ToolOutput,
                                      StreamEvent)
from finworld.models import (Model,
                        agglomerate_stream_deltas,
                        parse_json_if_needed)
from finworld.utils import (
    AgentImage,
    AgentAudio,
    AgentText,
)

from finworld.log import logger, YELLOW_HEX

class ToolCallingAgent(MultiStepAgent):
    """
    This agent uses JSON-like tool calls, using method `model.get_tool_call` to leverage the LLM engine's tool calling capabilities.

    Args:
        tools (`list[Tool]`): [`Tool`]s that the agent can use.
        model (`Model`): Model that will generate the agent's actions.
        prompt_templates ([`~agents.PromptTemplates`], *optional*): Prompt templates.
        planning_interval (`int`, *optional*): Interval at which the agent will run a planning step.
        stream_outputs (`bool`, *optional*, default `False`): Whether to stream outputs during execution.
        max_tool_threads (`int`, *optional*): Maximum number of threads for parallel tool calls.
            Higher values increase concurrency but resource usage as well.
            Defaults to `ThreadPoolExecutor`'s default.
        **kwargs: Additional keyword arguments.
    """

    def __init__(
        self,
        tools: list[Tool],
        model: Model,
        prompt_templates: PromptTemplates | None = None,
        planning_interval: int | None = None,
        stream_outputs: bool = False,
        max_tool_threads: int | None = None,
        **kwargs,
    ):
        prompt_templates = prompt_templates or yaml.safe_load(
            importlib.resources.files("src.base.prompts").joinpath("toolcalling_agent.yaml").read_text()
        )
        super().__init__(
            tools=tools,
            model=model,
            prompt_templates=prompt_templates,
            planning_interval=planning_interval,
            **kwargs,
        )
        # Streaming setup
        self.stream_outputs = stream_outputs
        if self.stream_outputs and not hasattr(self.model, "generate_stream"):
            raise ValueError(
                "`stream_outputs` is set to True, but the model class implements no `generate_stream` method."
            )
        # Tool calling setup
        self.max_tool_threads = max_tool_threads

    @property
    def tools_and_managed_agents(self):
        """Returns a combined list of tools and managed agents."""
        return list(self.tools.values()) + list(self.managed_agents.values())

    def initialize_system_prompt(self) -> str:
        system_prompt = populate_template(
            self.prompt_templates["system_prompt"],
            variables={
                "tools": self.tools,
                "managed_agents": self.managed_agents,
                "custom_instructions": self.instructions,
            },
        )
        return system_prompt

    def _step_stream(self, memory_step: ActionStep) -> Generator[ChatMessageStreamDelta | ToolOutput]:
        """
        Perform one step in the ReAct framework: the agent thinks, acts, and observes the result.
        Yields ChatMessageStreamDelta during the run if streaming is enabled.
        At the end, yields either None if the step is not final, or the final answer.
        """
        memory_messages = self.write_memory_to_messages()

        input_messages = memory_messages.copy()

        # Add new step in logs
        memory_step.model_input_messages = input_messages

        try:
            if self.stream_outputs and hasattr(self.model, "generate_stream"):
                output_stream = self.model.generate_stream(
                    input_messages,
                    stop_sequences=["Observation:", "Calling tools:"],
                    tools_to_call_from=self.tools_and_managed_agents,
                )

                chat_message_stream_deltas: list[ChatMessageStreamDelta] = []
                with Live("", console=self.logger.console, vertical_overflow="visible") as live:
                    for event in output_stream:
                        chat_message_stream_deltas.append(event)
                        live.update(
                            Markdown(agglomerate_stream_deltas(chat_message_stream_deltas).render_as_markdown())
                        )
                        yield event
                chat_message = agglomerate_stream_deltas(chat_message_stream_deltas)
            else:
                chat_message: ChatMessage = self.model.generate(
                    input_messages,
                    stop_sequences=["Observation:", "Calling tools:"],
                    tools_to_call_from=self.tools_and_managed_agents,
                )

                self.logger.log_markdown(
                    content=chat_message.content if chat_message.content else str(chat_message.raw),
                    title="Output message of the LLM:",
                    level=LogLevel.DEBUG,
                )

            # Record model output
            memory_step.model_output_message = chat_message
            memory_step.model_output = chat_message.content
            memory_step.token_usage = chat_message.token_usage
        except Exception as e:
            raise AgentGenerationError(f"Error while generating output:\n{e}", self.logger) from e

        if chat_message.tool_calls is None or len(chat_message.tool_calls) == 0:
            try:
                chat_message = self.model.parse_tool_calls(chat_message)
            except Exception as e:
                raise AgentParsingError(f"Error while parsing tool call from model output: {e}", self.logger)
        else:
            for tool_call in chat_message.tool_calls:
                tool_call.function.arguments = parse_json_if_needed(tool_call.function.arguments)
        yield from self.process_tool_calls(chat_message, memory_step)

    def process_tool_calls(self, chat_message: ChatMessage, memory_step: ActionStep) -> Generator[StreamEvent]:
        """Process tool calls from the model output and update agent memory.

        Args:
            chat_message (`ChatMessage`): Chat message containing tool calls from the model.
            memory_step (`ActionStep)`: Memory ActionStep to update with results.

        Yields:
            `ActionOutput`: The final output of tool execution.
        """
        model_outputs = []
        tool_calls = []
        observations = []

        final_answer_call = None
        parallel_calls = []
        assert chat_message.tool_calls is not None
        for tool_call in chat_message.tool_calls:
            yield tool_call
            tool_name = tool_call.function.name
            tool_arguments = tool_call.function.arguments
            model_outputs.append(str(f"Called Tool: '{tool_name}' with arguments: {tool_arguments}"))
            tool_calls.append(ToolCall(name=tool_name, arguments=tool_arguments, id=tool_call.id))
            # Track final_answer separately, add others to parallel processing list
            if tool_name == "final_answer":
                final_answer_call = (tool_name, tool_arguments)
                break  # Stop: final answer reached, no further tool calls
            else:
                parallel_calls.append((tool_name, tool_arguments))

        # Helper function to process a single tool call
        def process_single_tool_call(call_info):
            tool_name, tool_arguments = call_info
            self.logger.log(
                Panel(Text(f"Calling tool: '{tool_name}' with arguments: {tool_arguments}")),
                level=LogLevel.INFO,
            )
            if tool_arguments is None:
                tool_arguments = {}
            tool_call_result = self.execute_tool_call(tool_name, tool_arguments)
            tool_call_result_type = type(tool_call_result)
            if tool_call_result_type in [AgentImage, AgentAudio]:
                if tool_call_result_type == AgentImage:
                    observation_name = "image.png"
                elif tool_call_result_type == AgentAudio:
                    observation_name = "audio.mp3"
                # TODO: tool_call_result naming could allow for different names of same type
                self.state[observation_name] = tool_call_result
                observation = f"Stored '{observation_name}' in memory."
            else:
                observation = str(tool_call_result).strip()
            self.logger.log(
                f"Observations: {observation.replace('[', '|')}",  # escape potential rich-tag-like components
                level=LogLevel.INFO,
            )
            return observation

        # Process tool calls in parallel
        if parallel_calls:
            if len(parallel_calls) == 1:
                # If there's only one call, process it directly
                observations.append(process_single_tool_call(parallel_calls[0]))
                yield ToolOutput(output=None, is_final_answer=False)
            else:
                # If multiple tool calls, process them in parallel
                with ThreadPoolExecutor(self.max_tool_threads) as executor:
                    futures = [executor.submit(process_single_tool_call, call_info) for call_info in parallel_calls]
                    for future in as_completed(futures):
                        observations.append(future.result())
                        yield ToolOutput(output=None, is_final_answer=False)

        # Process final_answer call if present
        if final_answer_call:
            tool_name, tool_arguments = final_answer_call
            self.logger.log(
                Panel(Text(f"Calling tool: '{tool_name}' with arguments: {tool_arguments}")),
                level=LogLevel.INFO,
            )
            answer = (
                tool_arguments["answer"]
                if isinstance(tool_arguments, dict) and "answer" in tool_arguments
                else tool_arguments
            )
            if isinstance(answer, str) and answer in self.state.keys():
                # if the answer is a state variable, return the value
                # State variables are not JSON-serializable (AgentImage, AgentAudio) so can't be passed as arguments to execute_tool_call
                final_answer = self.state[answer]
                self.logger.log(
                    f"[bold {YELLOW_HEX}]Final answer:[/bold {YELLOW_HEX}] Extracting key '{answer}' from state to return value '{final_answer}'.",
                    level=LogLevel.INFO,
                )
            else:
                # Allow arbitrary keywords
                final_answer = self.execute_tool_call("final_answer", tool_arguments)
                self.logger.log(
                    Text(f"Final answer: {final_answer}", style=f"bold {YELLOW_HEX}"),
                    level=LogLevel.INFO,
                )
            memory_step.action_output = final_answer
            yield ToolOutput(output=final_answer, is_final_answer=True)

        # Update memory step with all results
        if model_outputs:
            memory_step.model_output = "\n".join(model_outputs)
        if tool_calls:
            memory_step.tool_calls = tool_calls
        if observations:
            memory_step.observations = "\n".join(observations)

    def _substitute_state_variables(self, arguments: dict[str, str] | str) -> dict[str, Any] | str:
        """Replace string values in arguments with their corresponding state values if they exist."""
        if isinstance(arguments, dict):
            return {
                key: self.state.get(value, value) if isinstance(value, str) else value
                for key, value in arguments.items()
            }
        return arguments

    def execute_tool_call(self, tool_name: str, arguments: dict[str, str] | str) -> Any:
        """
        Execute a tool or managed agent with the provided arguments.

        The arguments are replaced with the actual values from the state if they refer to state variables.

        Args:
            tool_name (`str`): Name of the tool or managed agent to execute.
            arguments (dict[str, str] | str): Arguments passed to the tool call.
        """
        # Check if the tool exists
        available_tools = {**self.tools, **self.managed_agents}
        if tool_name not in available_tools:
            raise AgentToolExecutionError(
                f"Unknown tool {tool_name}, should be one of: {', '.join(available_tools)}.", self.logger
            )

        # Get the tool and substitute state variables in arguments
        tool = available_tools[tool_name]
        arguments = self._substitute_state_variables(arguments)
        is_managed_agent = tool_name in self.managed_agents

        try:
            # Call tool with appropriate arguments
            if isinstance(arguments, dict):
                return tool(**arguments) if is_managed_agent else tool(**arguments, sanitize_inputs_outputs=True)
            elif isinstance(arguments, str):
                return tool(arguments) if is_managed_agent else tool(arguments, sanitize_inputs_outputs=True)
            else:
                raise TypeError(f"Unsupported arguments type: {type(arguments)}")

        except TypeError as e:
            # Handle invalid arguments
            description = getattr(tool, "description", "No description")
            if is_managed_agent:
                error_msg = (
                    f"Invalid request to team member '{tool_name}' with arguments {json.dumps(arguments)}: {e}\n"
                    "You should call this team member with a valid request.\n"
                    f"Team member description: {description}"
                )
            else:
                error_msg = (
                    f"Invalid call to tool '{tool_name}' with arguments {json.dumps(arguments)}: {e}\n"
                    "You should call this tool with correct input arguments.\n"
                    f"Expected inputs: {json.dumps(tool.inputs)}\n"
                    f"Returns output type: {tool.output_type}\n"
                    f"Tool description: '{description}'"
                )
            raise AgentToolCallError(error_msg, self.logger) from e

        except Exception as e:
            # Handle execution errors
            if is_managed_agent:
                error_msg = (
                    f"Error executing request to team member '{tool_name}' with arguments {json.dumps(arguments)}: {e}\n"
                    "Please try again or request to another team member"
                )
            else:
                error_msg = (
                    f"Error executing tool '{tool_name}' with arguments {json.dumps(arguments)}: {type(e).__name__}: {e}\n"
                    "Please try again or use another tool"
                )
            raise AgentToolExecutionError(error_msg, self.logger) from e