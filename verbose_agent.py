from typing import TypeVar, AsyncGenerator, Optional
from google.adk.agents import BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events.event import Event
from google.genai import types

AgentType = TypeVar("AgentType", bound=BaseAgent)
def VerboseAgent(
        agent: AgentType,
        before_agent_message: Optional[str] = "",
        after_agent_message: Optional[str] = None,
    ) -> AgentType:
    
    async def _run_async_impl(ctx: InvocationContext) -> AsyncGenerator[Event, None]:
        if before_agent_message is not None:
            event = Event(
                partial=True,
                author="system",
                content=types.Content(role="model", parts=[types.Part(text=f"[{ctx.agent.name}]: {before_agent_message or 'Started Running'}")])
            )
            yield event

        async for event in ctx.agent.__class__._run_async_impl(ctx.agent, ctx):
            yield event

        if after_agent_message is not None:
            event = Event(
                partial=True,
                author="system",
                content=types.Content(role="model", parts=[types.Part(text=f"[{ctx.agent.name}]: {after_agent_message or 'Done Running'}")])
            )
            yield event
        
    agent.__dict__["_run_async_impl"] = _run_async_impl
    return agent