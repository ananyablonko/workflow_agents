import typing
import asyncio
from typing import AsyncGenerator, Callable, Any, Optional, ClassVar
from typing_extensions import override
from google.adk.agents import BaseAgent, SequentialAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event, EventActions
from google.adk.agents.parallel_agent import _merge_agent_run
from google.genai.types import Content, Part

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
    state_callback: Optional[
        Callable[[InvocationContext], Any]
    ]

    def __init__(self,
                name: str,
                agent: BaseAgent,
                input_key: str,
                output_key: str,
                description: str = '',
                state_callback: Optional[
                    Callable[[InvocationContext], Any]
                ] = None,
                ):
        super().__init__(
            name=name, description=description, sub_agents=[agent],
            agent=agent, input_key=input_key, output_key=output_key, state_callback=state_callback  # type: ignore
        )  

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

        res = [None] * len(prompts)
        async for event in _merge_agent_run([ctx.agent.run_async(ctx) for ctx in contexts]):
            yield event
            if event.actions and event.actions.state_delta and event.branch:
                idx = int(event.branch.split(self.branch_prefix)[-1].split('.')[0])
                ctx = contexts[idx]
                res[idx] = (
                    self.state_callback(ctx) if self.state_callback is not None else ctx.session.state.get(get_output_key(self.agent))
                )

        event = Event(
            author=self.name,
            actions=EventActions(state_delta={self.output_key: res}),
        )
        await invocation_context.session_service.append_event(invocation_context.session, event)
        

    async def _branch_context(self, old_ctx: InvocationContext, *, branch_idx: int, prompt: str) -> InvocationContext:
        branch_name = f"{old_ctx.branch}.{self.branch_prefix}{branch_idx}"
        ctx = old_ctx.model_copy(
            update=dict(
                branch=branch_name,
                agent=self.agent
            )
        )
        event = Event(
            author="user",
            branch=branch_name,
            content=Content(
                role="user",
                parts=[Part(text=prompt)],
            ),
        )
        await ctx.session_service.append_event(ctx.session, event)
        return ctx
