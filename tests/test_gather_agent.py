import __init__
import asyncio

import logging, sys
logger = logging.Logger(__name__)
logging.basicConfig(level=logging.INFO, stream=sys.stdout)
from pydantic import RootModel
from google.adk.agents import LlmAgent, SequentialAgent
from google.adk.sessions import Session

from syllabus_agent.worflow_agents.gather_agent import GatherAgent
from utils.run_agent import run_agent

async def test_gather_agent():
    class ListModel(RootModel[list[str]]):
        pass

    splitter = LlmAgent(
        name="pattern_continuist",
        model="gemini-2.0-flash",
        instruction = """
    Continue the input into a 3 item pattern.
    "A" -> ["A","B","C"]
    "Y" -> ["Y","Z","A"]
    "5" -> ["5","6","7"]
    """,
        output_schema=ListModel,
        output_key='list',
        disallow_transfer_to_parent=True,
        disallow_transfer_to_peers=True,
    )

    assert splitter.output_key

    gather = GatherAgent(
        name="TestGatherAgent",
        agent=splitter.model_copy(update={'name': 'pattern_continuist_2'}),
        input_key="list",
        output_key="gather_output",
    )

    app_agent = SequentialAgent(name='TestAgent', sub_agents=[splitter, gather])

    start = '1'
    expected_output = {('1', '2', '3'), ('2', '3', '4'), ('3', '4', '5')}

    async for state_delta in run_agent(
        app_agent,
        start,
        check = lambda e: True,
        func = lambda e: e.actions.state_delta
    ):
        if isinstance(state_delta, Session):
            break
        assert splitter.output_key in state_delta
        splitter_output = state_delta[splitter.output_key]
        assert isinstance(splitter_output, list)
        assert tuple(splitter_output) in expected_output

    session = state_delta

    assert session is not None
    res = session.state.get(gather.output_key)
    assert isinstance(res, dict)
    assert set([tuple(x) for x in res.values()]) == expected_output



def main() -> None:
    asyncio.run(test_gather_agent())


if __name__ == '__main__':
    main()