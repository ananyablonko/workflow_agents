import asyncio
from pydantic import RootModel, Field, PrivateAttr
from typing import AsyncGenerator, Optional
from typing_extensions import override
from google.adk.agents import BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event, EventActions
from google.adk.agents.parallel_agent import _merge_agent_run
from google.genai import types

from google.adk.models.google_llm import logger


class GatherAgent(BaseAgent):
    sub_agents: list[BaseAgent] = Field(min_length=1, max_length=1, default_factory=list)
    output_key: Optional[str] = None
    input_key: Optional[str] = None
    _key_agents: set[str] = PrivateAttr(default_factory=set)

    @property
    def agent(self):
        return self.sub_agents[0]

    @override
    async def _run_async_impl(
        self, invocation_context: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        invocation_context.branch = invocation_context.branch or self.name
        prompts: list[str]

        # Use input_key to get prompts from session state if available
        # Otherwise, extract the last user input from session events
        if self.input_key and self.input_key in invocation_context.session.state:
            agent_input = invocation_context.session.state.get(self.input_key, [])
            prompts = RootModel[list[str]].model_validate(agent_input).model_dump()
        else:
            i, agent_input = next(
                (
                    (i, e.content.parts[0].text)
                    for i, e in enumerate(reversed(invocation_context.session.events))
                    if e.content and e.content.parts and e.content.parts[0].text
                ), (0, '[]'),
            )
            prompts = RootModel[list[str]].model_validate_json(agent_input).model_dump()
            invocation_context.session.events.pop(-i-1)  # removed only for this agent call

        w = len(str(len(prompts)))  # for agent naming

        contexts = await asyncio.gather(*[
            self._branch_context(invocation_context, idx=i, prompt=prompt, width=w)
            for i, prompt in enumerate(prompts)
        ])

        async for event in _merge_agent_run([ctx.agent.run_async(ctx) for ctx in contexts]):
            yield event

        res = {
            name: [ctx.session.state.get(self._get_unique_name(i, w, name), None) for i, ctx in enumerate(contexts)]
            for name in self._key_agents
        }

        if self.output_key:
            yield Event(
                invocation_id=invocation_context.invocation_id,
                branch=invocation_context.branch,
                author=self.name,
                actions=EventActions(state_delta={self.output_key: res}),
            )

        self._key_agents = set()

    @staticmethod
    def _get_unique_name(idx: int, width: int, name: str) -> str:
        return f"{name}_{idx:0{width}d}"

    async def _branch_context(self, old_ctx: InvocationContext, *, idx: int, prompt: str, width: int) -> InvocationContext:
        # rename the agent to have the branch idx in its name (and consequently its branch), and copy user contents to allow appending to it in branch only
        agent = self._rename_agent_tree(self.agent, idx, width)
        branch = f"{old_ctx.branch}.{agent.name}"
        prompt_part = [types.Part(text=prompt)]

        ctx = old_ctx.model_copy(
            update=dict(
                branch=branch,
                agent=agent,
                # For easy availability of agent input in callbacks
                user_content=types.Content(role="user", parts=((old_ctx.user_content or types.Content()).parts or []) + prompt_part),
            )
        )

        await ctx.session_service.append_event(
            ctx.session,
            Event(
                author="user",
                branch=branch,  # event only visible to one instance
                content=types.Content(
                    role="user",
                    parts=prompt_part,
                ),
            ),
        )

        return ctx

    def _rename_agent_tree(self, agent: BaseAgent, idx: int, width: int, inside_gather_agent: bool = False) -> BaseAgent:
        is_gather = isinstance(agent, GatherAgent)
        new_agent = agent.model_copy(update=dict(_key_agents=set()) if is_gather else {})
        new_name = self._get_unique_name(idx, width, new_agent.name)
        if hasattr(new_agent, "output_key") and not inside_gather_agent:
            if getattr(new_agent, "output_key", None) is not None:
                logger.warning(
                    f"Overriding agent {new_agent.name} output_key. Normal state delta is not thread-safe, so regular output_key should not be used in GatherAgent sub-agents."
                )
            self._key_agents.add(agent.name)
            setattr(new_agent, "output_key", new_name)
        new_agent.name = new_name

        new_agent.sub_agents = [
            self._rename_agent_tree(a, idx, width, inside_gather_agent=inside_gather_agent or is_gather)
            for a in agent.sub_agents
        ]
        return new_agent
