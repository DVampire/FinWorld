from typing import Dict, List, Optional, Any
from collections.abc import Generator
from openai.types.chat import ChatCompletion

from finworld.models.llm_provider.base import (ApiModel,
                             ChatMessage,
                             tool_role_conversions,
                             ChatMessageStreamDelta,
                             ChatMessageToolCallStreamDelta)
from finworld.models.llm_provider.message_manager import MessageManager
from finworld.log import TokenUsage


class RestfulClient():
    def __init__(self,
                 api_base: str,
                 api_key: str,
                 api_type: str = "chat/completions",
                 model_id: str = "o3",
                 http_client=None):
        self.api_base = api_base
        self.api_key = api_key
        self.api_type = api_type
        self.model_id = model_id

        self.http_client = http_client

    def completion(self,
                   model,
                   messages,
                   **kwargs):

        headers = {
            "app_key": self.api_key,
            "Content-Type": "application/json"
        }

        model = model.split("/")[-1]
        data = {
            "model": model,
            "messages": messages,
        }

        # Add any additional kwargs to the data
        if kwargs:
            data.update(kwargs)

        response = self.http_client.post(
            f"{self.api_base}/{self.api_type}",
            json=data,
            headers=headers,
        )

        return response.json()

    async def acompletion(self,
                          model,
                          messages,
                          **kwargs):

        headers = {
            "app_key": self.api_key,
            "Content-Type": "application/json"
        }

        model = model.split("/")[-1]
        data = {
            "model": model,
            "messages": messages,
        }

        # Add any additional kwargs to the data
        if kwargs:
            data.update(kwargs)

        response = await self.http_client.post(
            f"{self.api_base}/{self.api_type}",
            json=data,
            headers=headers,
        )

        return response.json()


class RestfulModel(ApiModel):
    """This model connects to an OpenAI-compatible API server.

    Parameters:
        model_id (`str`):
            The model identifier to use on the server (e.g. "gpt-3.5-turbo").
        api_base (`str`, *optional*):
            The base URL of the OpenAI-compatible API server.
        api_key (`str`, *optional*):
            The API key to use for authentication.
        organization (`str`, *optional*):
            The organization to use for the API request.
        project (`str`, *optional*):
            The project to use for the API request.
        client_kwargs (`dict[str, Any]`, *optional*):
            Additional keyword arguments to pass to the OpenAI client (like organization, project, max_retries etc.).
        custom_role_conversions (`dict[str, str]`, *optional*):
            Custom role conversion mapping to convert message roles in others.
            Useful for specific models that do not support specific message roles like "system".
        flatten_messages_as_text (`bool`, default `False`):
            Whether to flatten messages as text.
        **kwargs:
            Additional keyword arguments to pass to the OpenAI API.
    """

    def __init__(
        self,
        model_id: str,
        api_base: Optional[str] = None,
        api_key: Optional[str] = None,
        custom_role_conversions: dict[str, str] | None = None,
        flatten_messages_as_text: bool = False,
        http_client=None,
        **kwargs,
    ):
        self.model_id = model_id
        self.api_base = api_base
        self.api_key = api_key
        flatten_messages_as_text = (
            flatten_messages_as_text
            if flatten_messages_as_text is not None
            else model_id.startswith(("ollama", "groq", "cerebras"))
        )

        self.http_client = http_client

        self.message_manager = MessageManager(model_id=model_id)

        super().__init__(
            model_id=model_id,
            custom_role_conversions=custom_role_conversions,
            flatten_messages_as_text=flatten_messages_as_text,
            **kwargs,
        )

    def create_client(self):
        return RestfulClient(api_base=self.api_base,
                             api_key=self.api_key,
                             model_id=self.model_id,
                             http_client=self.http_client)

    def _prepare_completion_kwargs(
            self,
            messages: list[ChatMessage],
            stop_sequences: list[str] | None = None,
            response_format: dict[str, str] | None = None,
            tools_to_call_from: list[Any] | None = None,
            custom_role_conversions: dict[str, str] | None = None,
            convert_images_to_image_urls: bool = False,
            tool_choice: str | dict | None = "required",  # Configurable tool_choice parameter
            **kwargs,
    ) -> dict[str, Any]:
        """
        Prepare parameters required for model invocation, handling parameter priorities.

        Parameter priority from high to low:
        1. Explicitly passed kwargs
        2. Specific parameters (stop_sequences, response_format, etc.)
        3. Default values in self.kwargs
        """
        # Clean and standardize the message list
        flatten_messages_as_text = kwargs.pop("flatten_messages_as_text", self.flatten_messages_as_text)
        messages_as_dicts = self.message_manager.get_clean_message_list(
            messages,
            role_conversions=custom_role_conversions or tool_role_conversions,
            convert_images_to_image_urls=convert_images_to_image_urls,
            flatten_messages_as_text=flatten_messages_as_text,
        )
        # Use self.kwargs as the base configuration
        completion_kwargs = {
            **self.kwargs,
            "messages": messages_as_dicts,
        }

        # Handle specific parameters
        if stop_sequences is not None:
            completion_kwargs["stop"] = stop_sequences
        if response_format is not None:
            completion_kwargs["response_format"] = response_format

        # Handle tools parameter
        if tools_to_call_from:
            tools_config = {
                "tools": [self.message_manager.get_tool_json_schema(tool, model_id=self.model_id) for tool in
                          tools_to_call_from],
            }
            if tool_choice is not None:
                tools_config["tool_choice"] = tool_choice
            completion_kwargs.update(tools_config)

        # Finally, use the passed-in kwargs to override all settings
        completion_kwargs.update(kwargs)

        completion_kwargs = self.message_manager.get_clean_completion_kwargs(completion_kwargs)

        return completion_kwargs

    def generate_stream(self,
                        messages: list[ChatMessage],
                        stop_sequences: list[str] | None = None,
                        response_format: dict[str, str] | None = None,
                        tools_to_call_from: list[Any] | None = None,
                        **kwargs,
                        )-> Generator[ChatMessageStreamDelta]:

        completion_kwargs = self._prepare_completion_kwargs(
            messages=messages,
            stop_sequences=stop_sequences,
            response_format=response_format,
            tools_to_call_from=tools_to_call_from,
            model=self.model_id,
            custom_role_conversions=self.custom_role_conversions,
            convert_images_to_image_urls=True,
            **kwargs,
        )

        for event in self.client.completion(**completion_kwargs, stream=True, stream_options={"include_usage": True}):
            if getattr(event, "usage", None):
                self._last_input_token_count = event.usage.prompt_tokens
                self._last_output_token_count = event.usage.completion_tokens
                yield ChatMessageStreamDelta(
                    content="",
                    token_usage=TokenUsage(
                        input_tokens=event.usage.prompt_tokens,
                        output_tokens=event.usage.completion_tokens,
                    ),
                )
            if event.choices:
                choice = event.choices[0]
                if choice.delta:
                    yield ChatMessageStreamDelta(
                        content=choice.delta.content,
                        tool_calls=[
                            ChatMessageToolCallStreamDelta(
                                index=delta.index,
                                id=delta.id,
                                type=delta.type,
                                function=delta.function,
                            )
                            for delta in choice.delta.tool_calls
                        ]
                        if choice.delta.tool_calls
                        else None,
                    )
                else:
                    if not getattr(choice, "finish_reason", None):
                        raise ValueError(f"No content or tool calls in event: {event}")


    async def generate(
        self,
        messages: list[ChatMessage],
        stop_sequences: list[str] | None = None,
        response_format: dict[str, str] | None = None,
        tools_to_call_from: list[Any] | None = None,
        **kwargs,
    ) -> ChatMessage:

        completion_kwargs = self._prepare_completion_kwargs(
            messages=messages,
            stop_sequences=stop_sequences,
            response_format=response_format,
            tools_to_call_from=tools_to_call_from,
            model=self.model_id,
            convert_images_to_image_urls=True,
            custom_role_conversions=self.custom_role_conversions,
            **kwargs,
        )

        # Async call to the LiteLLM client for completion
        response = await self.client.acompletion(**completion_kwargs)

        response = ChatCompletion.model_validate(response)

        self._last_input_token_count = response.usage.prompt_tokens
        self._last_output_token_count = response.usage.completion_tokens
        return ChatMessage.from_dict(
            response.choices[0].message.model_dump(include={"role", "content", "tool_calls"}),
            raw=response,
            token_usage=TokenUsage(
                input_tokens=response.usage.prompt_tokens,
                output_tokens=response.usage.completion_tokens,
            ),
        )

    async def __call__(self, *args, **kwargs) -> ChatMessage:
        """
        Call the model with the given arguments.
        This is a convenience method that calls `generate` with the same arguments.
        """
        return await self.generate(*args, **kwargs)