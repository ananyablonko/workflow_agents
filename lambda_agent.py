from typing import Callable, Coroutine
from google.adk.agents import BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event

class LambdaAgent(BaseAgent):
    func: Callable[[InvocationContext], Coroutine[None, None, Event]]

    async def _run_async_impl(self, ctx: InvocationContext):
        yield await self.func(ctx)

