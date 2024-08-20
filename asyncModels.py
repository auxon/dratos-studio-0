class VLLMAsync(VLLM):
    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that python package exists in environment."""
        from vllm import AsyncEngineArgs

        values["client"] = AsyncLLMEngine.from_engine_args(
            engine_args=AsyncEngineArgs(
                model=values["model"],
                trust_remote_code=True,
                download_dir=values["download_dir"],
                max_model_len=values["vllm_kwargs"]["max_model_len"],
                seed=values["vllm_kwargs"]["seed"],
            ),
        )

        return values

    async def _agenerate(
        self,
        prompts: List[str],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        """Run the LLM on the given prompt and input."""

        # build sampling parameters
        params = {**self._default_params, **kwargs, "stop": stop}
        sampling_params = SamplingParams(**params)
        # call the model
        client = self.client  # type: AsyncLLMEngine

        # generations: List[ChatGeneration] = []
        for prompt in prompts:
            output: RequestOutput
            async for output in client.generate(
                prompt=prompt, sampling_params=sampling_params, request_id=random_uuid()
            ):
                text = output.outputs[0].text
                output: CompletionOutput = output.outputs[0]
                # generation_info = output.__dict__
                # generations.append(
                #     ChatGenerationChunk(
                #         message=AIMessage(content=text),
                #         generation_info=generation_info,
                #     )
                # )
                if output:
                    yield ChatGenerationChunk(
                        message=AIMessageChunk(content=output.outputs[0].text),
                        generation_info=output.outputs[0].generation_info,
                    )
                    text = output.outputs[0].text
                    # generation_info = output.outputs[0].generation_info
                    if run_manager:
                        await run_manager.on_llm_new_token(text, verbose=self.verbose)

    async def _astream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        """Stream text generation asynchronously.

        Args:
            prompt: The prompt to pass into the model.
            stop: Optional list of stop words to use when generating.

        Yields:
            GenerationChunk: Generated text chunks.
        """
        prompt = self._format_messages_as_text(messages)

        # build sampling parameters
        params = {**self._default_params, **kwargs, "stop": stop}
        sampling_params = SamplingParams(**params)
        # call the model
        client = self.client  # type: AsyncLLMEngine

        async for output in client.generate(prompt, sampling_params):
            if output:
                yield ChatGenerationChunk(
                    message=AIMessageChunk(content=output.outputs[0].text),
                    generation_info=output.outputs[0].generation_info,
                )
                text = output.outputs[0].text
                if run_manager:
                    await run_manager.on_llm_new_token(text, verbose=self.verbose)

    def _format_message_as_text(self, message: BaseMessage) -> str:
        if isinstance(message, ChatMessage):
            message_text = f"\n\n{message.role.capitalize()}: {message.content}"
        elif isinstance(message, HumanMessage):
            message_text = f"[INST] {message.content} [/INST]"
        elif isinstance(message, AIMessage):
            message_text = f"{message.content}"
        elif isinstance(message, SystemMessage):
            message_text = f"<<SYS>> {message.content} <</SYS>>"
        else:
            raise ValueError(f"Got unknown type {message}")
        return message_text

    def _format_messages_as_text(self, messages: List[BaseMessage]) -> str:
        return "\n".join(
            [self._format_message_as_text(message) for message in messages]
        )