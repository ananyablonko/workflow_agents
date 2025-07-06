from __future__ import annotations
from typing import AsyncGenerator
from typing_extensions import override

from google.adk.agents import BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event

from .gather_agent import GatherAgent
from .utils.adk.state import get_output_key


class RecursiveAgent(BaseAgent):
    evaluator: BaseAgent
    splitter: BaseAgent
    input_key: str
    max_depth: int
    evaluator_output_key: str
    splitter_output_key: str

    def __init__(
        self,
        *,
        evaluator: BaseAgent,
        splitter: BaseAgent,
        input_key: str,
        name: str,
        description: str = '',
        max_depth: int = 3,
    ):
        evaluator_output_key = get_output_key(evaluator)
        splitter_output_key = get_output_key(splitter)

        super().__init__(
            evaluator=evaluator,        # type: ignore
            splitter=splitter,          # type: ignore
            input_key=input_key,        # type: ignore
            name=name,
            description=description,
            max_depth=max_depth,        # type: ignore
            evaluator_output_key=evaluator_output_key, # type: ignore
            splitter_output_key=splitter_output_key, # type: ignore
            sub_agents=[evaluator, splitter],
        )


    # ------------------------------------------------------------------ #
    @override
    async def _run_async_impl(
        self, ctx: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        # ------------ 1. evaluate ------------------------------------- #
        eval_ctx = ctx.model_copy(
            update=dict(
                agent=self.evaluator,
                branch=f"{ctx.branch}.eval" if ctx.branch else "eval",
            )
        )
        async for e in self.evaluator.run_async(eval_ctx):
            yield e

        passed = eval_ctx.session.state[self.evaluator_output_key]["output"]
        if not passed or self.max_depth <= 0:
            return  # stop recursion

        # ------------ 2. split ---------------------------------------- #
        split_ctx = ctx.model_copy(
            update=dict(
                agent=self.splitter,
                branch=f"{ctx.branch}.split" if ctx.branch else "split",
            )
        )
        async for e in self.splitter.run_async(split_ctx):
            yield e


        # ------------ 3. recurse via GatherAgent ---------------------- #
        gather = GatherAgent(
            name=f"{self.name}_gather_d{self.max_depth}",
            sub_agents=[self.model_copy(update=dict(name=f"{self.name}_d{self.max_depth-1}", max_depth=self.max_depth - 1))],
            input_key=self.input_key,
            output_key=f"{self.name}_gather_d{self.max_depth}"
        )

        gather_ctx = ctx.model_copy(
            update=dict(
                agent=gather,
                branch=f"{ctx.branch}.gather" if ctx.branch else "gather",
            )
        )

        async for event in gather.run_async(gather_ctx):
            yield event
