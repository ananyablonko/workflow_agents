import asyncio
from pydantic import BaseModel
from typing import AsyncGenerator, Optional, Any
from typing_extensions import override
from google.adk.agents import BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event, EventActions
from google.adk.agents.parallel_agent import _merge_agent_run
from google.genai import types


class GatherAgent(BaseAgent):
    input_key: str
    output_key: str
    key_agent_name: str | None = None

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

        async for event in _merge_agent_run([ctx.agent.run_async(ctx) for ctx in contexts]):
            yield event

        res = [ctx.session.state.get(self.get_unique_name(i, w, self.key_agent_name), None) for i, ctx in enumerate(contexts)]
        
        event = Event(
            invocation_id=invocation_context.invocation_id,
            branch=invocation_context.branch,
            author=self.name,
            actions=EventActions(state_delta={self.output_key: res}),
        )
        yield event
    
    def get_unique_name(self, idx: int, width: int, name: Optional[str] = None) -> str:
        return f"{name or self.agent.name}_{idx:0{width}d}"

    async def _branch_context(self, old_ctx: InvocationContext, *, idx: int, prompt: Any, width: int) -> InvocationContext:
        # rename the agent to have the branch idx in its name (and consequently its branch), and copy user contents to allow appending to it in branch only
        
        agent = self.rename_agent_tree(self.agent, idx, width)
        branch = f"{old_ctx.branch}.{agent.name}"

        text_part = types.Part(text=prompt.model_dump_json() if isinstance(prompt, BaseModel) else str(prompt))
        event = Event(
            author="user",
            branch=branch,  # event only visible to one instance
            content=types.Content(
                role="user",
                parts=[text_part],
            ),
        )

        ctx = old_ctx.model_copy(
            update=dict(
                branch=branch, agent=agent,
                user_content=types.Content(role="user", parts=((old_ctx.user_content or types.Content()).parts or []) + [text_part]),
            )
        )
       
        await ctx.session_service.append_event(ctx.session, event)
        return ctx
    
    def rename_agent_tree(self, agent: BaseAgent, idx: int, width: int) -> BaseAgent:
        new_agent = agent.model_copy()
        new_name = self.get_unique_name(idx, width, new_agent.name)
        if new_agent.name == (self.key_agent_name or self.agent.name):
            setattr(new_agent, "output_key", new_name)
        new_agent.name = new_name
        
        new_agent.sub_agents = [self.rename_agent_tree(a, idx, width) for a in agent.sub_agents]
        return new_agent
