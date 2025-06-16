import typing
import asyncio
from typing import AsyncGenerator, ClassVar, Optional
from typing_extensions import override
from google.adk.agents import BaseAgent, SequentialAgent, LoopAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event, EventActions
from google.adk.agents.parallel_agent import _merge_agent_run
from google.genai.types import Content, Part
from google.genai import types
from copy import deepcopy

import re

def get_output_key(agent: BaseAgent) -> str:
    key = None
    if hasattr(agent, 'output_key'):
        key = agent.output_key  # type: ignore
    elif isinstance(agent, SequentialAgent):
        key = get_output_key(agent.sub_agents[-1])
    else:
        raise NotImplementedError(f"{get_output_key.__name__} is only valid for Agents with an output key")
    if key is None:
        raise ValueError("Agent does not have an output key")
    return key


class GatherAgent(BaseAgent):
    branch_prefix: ClassVar[str] = '_gather_branch_'

    agent: BaseAgent
    input_key: str
    output_key: str

    def __init__(self,
                name: str,
                agent: BaseAgent,
                input_key: str,
                output_key: str,
                description: str = '',
                agent_output_key: Optional[str] = None,
                **kwargs
                ):
        super().__init__(
            name=name, description=description, sub_agents=[agent],
            agent=agent, input_key=input_key, output_key=output_key  # type: ignore
            **kwargs
        )  

        self._agent_output_key = agent_output_key or get_output_key(agent)

    @override
    async def _run_async_impl(
        self, invocation_context: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        invocation_context.branch = f"{invocation_context.branch}.{self.name}" if invocation_context.branch else self.name
        prompts: list[str] | dict = invocation_context.session.state.get(self.input_key, [])
        if isinstance(prompts, dict):
            prompts = typing.cast(list[str], list[prompts.values()])

        contexts = await asyncio.gather(*[
            self._branch_context(
                invocation_context,
                branch_idx=i,
                prompt=prompt,
            )
            for i, prompt in enumerate(prompts)
        ])

        res: list = [None] * len(prompts)
        async for event in _merge_agent_run([ctx.agent.run_async(ctx) for ctx in contexts]):
            yield event
            if event.actions and self._agent_output_key in event.actions.state_delta and event.branch:
                idx = int(re.findall(fr'(?<={self.branch_prefix})\d+', event.branch)[0])
                res[idx] = deepcopy(event.actions.state_delta[self._agent_output_key])

        event = Event(
            author=self.name,
            actions=EventActions(state_delta={self.output_key: res}),
        )
        await invocation_context.session_service.append_event(invocation_context.session, event)
        

    async def _branch_context(self, old_ctx: InvocationContext, *, branch_idx: int, prompt: str) -> InvocationContext:
        branch_name = f"{old_ctx.branch}.{self.branch_prefix}{branch_idx}"
        event = Event(
            author="user",
            branch=branch_name,
            content=Content(
                role="user",
                parts=[Part(text=prompt)],
            ),
        )
        ctx = old_ctx.model_copy(
            update=dict(
                branch=branch_name,
                agent=self.agent.model_copy(update=dict(name=f"{self.agent.name}_{branch_idx}")),
                user_content=old_ctx.user_content.model_copy(deep=True) if old_ctx.user_content else types.Content(role="user", parts=[]),
            )
        )
        
        if ctx.user_content is not None and ctx.user_content.parts is not None:
            ctx.user_content.parts.append(types.Part(text=prompt))
       
        await ctx.session_service.append_event(ctx.session, event)
        return ctx
