import typing
from typing import AsyncGenerator, Callable, Any, Optional
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
    name: str
    description: str = ''
    agent: BaseAgent
    input_key: str
    output_key: str
    state_callback: Optional[
        Callable[[InvocationContext], Any]
    ] = None

    @override
    async def _run_async_impl(
        self, invocation_context: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        invocation_context.branch = f"{invocation_context.branch}.{self.name}" if invocation_context.branch else self.name
        prompts: list[str] | dict = invocation_context.session.state.get(self.input_key, [])
        if isinstance(prompts, dict):
            prompts = typing.cast(list[str], list[prompts.values()])

        runs = []
        for i, prompt in enumerate(prompts):
            branch = f"{invocation_context.branch}._gather_branch_{i}"
            ctx = invocation_context.model_copy(
                update=dict(
                    branch=branch,
                    session=invocation_context.session.model_copy(),
                    agent=self.agent
                )
            )
            event = Event(
                author="user",
                branch=branch,
                content=Content(
                    role="user",
                    parts=[Part(text=prompt)],
                ),
            )
            await ctx.session_service.append_event(ctx.session, event)
            runs.append(ctx.agent.run_async(ctx))

        res = {}
        async for event in _merge_agent_run(runs):
            yield event
            if event.actions and event.actions.state_delta:
                res[event.branch] = (
                    self.state_callback(ctx) if self.state_callback is not None else ctx.session.state.get(get_output_key(self.agent))
                )

        event = Event(
            author=self.name,
            actions=EventActions(state_delta={self.output_key: res}),
        )
        await invocation_context.session_service.append_event(invocation_context.session, event)
