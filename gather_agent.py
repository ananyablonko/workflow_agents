import asyncio
from pydantic import BaseModel
from typing import AsyncGenerator, Optional, Any
from typing_extensions import override
from google.adk.agents import BaseAgent, LlmAgent, ParallelAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event, EventActions
from google.adk.agents.parallel_agent import _merge_agent_run
from google.genai.types import Content, Part
from google.genai import types
from copy import deepcopy

import re

def get_output_key(agent: BaseAgent) -> str:
    key = None
    if isinstance(agent, LlmAgent):
        key = agent.output_key
    elif agent.sub_agents and not isinstance(agent, ParallelAgent):
        key = get_output_key(agent.sub_agents[-1])
    else:
        raise NotImplementedError(f"{get_output_key.__name__} is only valid for Agents with an output key")
    if key is None:
        raise ValueError("Agent does not have an output key")
    return key


class GatherAgent(BaseAgent):
    input_key: str
    output_key: str
    key_agent_name: str

    @property
    def agent(self):
        return self.sub_agents[0]
    
    @override
    async def _run_async_impl(
        self, invocation_context: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        invocation_context.branch = invocation_context.branch or self.name
        prompts: list[Any] = invocation_context.session.state.get(self.input_key, [])

        w = len(str(len(prompts)))

        contexts = await asyncio.gather(*[
            self._branch_context(
                invocation_context,
                idx=i,
                prompt=prompt,
                width=w
            )
            for i, prompt in enumerate(prompts)
        ])

        res: list = [None for _ in range(len(prompts))]
        keys = set()
        async for event in _merge_agent_run([ctx.agent.run_async(ctx) for ctx in contexts]):
            if event.branch and event.author.startswith(self.key_agent_name) and event.author in event.actions.state_delta:
                idx = int(re.findall(fr'(?<={self.agent.name}_)\d+', event.branch)[-1])
                res[idx] = deepcopy(event.actions.state_delta[event.author])
                keys.add(event.author)
            yield event

        for key in keys:
            invocation_context.session.state.pop(key, None)
        
        event = Event(
            invocation_id=invocation_context.invocation_id,
            branch=invocation_context.branch,
            author=self.name,
            content=types.Content(role='model', parts=[Part(text=str(res))]),
            actions=EventActions(state_delta={self.output_key: res}),
        )
        yield event
    
    def get_unique_name(self, idx: int, width: int, name: Optional[str] = None) -> str:
        return f"{name or self.agent.name}_{idx:0{width}d}"

    async def _branch_context(self, old_ctx: InvocationContext, *, idx: int, prompt: Any, width: int) -> InvocationContext:
        # rename the agent to have the branch idx in its name (and consequently its branch), and copy user contents to allow appending to it in branch only

        ctx = old_ctx.model_copy(
            update=dict(
                agent=self.rename_agent_tree(self.agent, idx, width),
                user_content=old_ctx.user_content.model_copy(deep=True) if old_ctx.user_content else types.Content(role="user", parts=[]),
            )
        )
        text: str = prompt.model_dump_json() if isinstance(prompt, BaseModel) else str(prompt)
        event = Event(
            author="user",
            branch=f"{old_ctx.branch}.{ctx.agent.name}",  # event only visible to one instance
            content=Content(
                role="user",
                parts=[Part(text=text)],
            ),
        )
        
        if ctx.user_content is not None and ctx.user_content.parts is not None:
            ctx.user_content.parts.append(types.Part(text=text))
       
        await ctx.session_service.append_event(ctx.session, event)
        return ctx
    
    def rename_agent_tree(self, agent: BaseAgent, idx: int, width: int) -> BaseAgent:
        new_agent = agent.model_copy()
        new_name = self.get_unique_name(idx, width, new_agent.name)
        if new_agent.name == self.key_agent_name:
            setattr(new_agent, "output_key", new_name)
        new_agent.name = new_name
        
        new_agent.sub_agents = [self.rename_agent_tree(a, idx, width) for a in agent.sub_agents]
        return new_agent
