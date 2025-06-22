import __init__
import asyncio
import dotenv
dotenv.load_dotenv()
import logging, sys
logger = logging.Logger(__name__)
logging.basicConfig(level=logging.INFO, stream=sys.stdout)
from pydantic import RootModel
from google.adk.agents import LlmAgent, SequentialAgent
from agent_tester import AgentTester
from gather_agent import GatherAgent
from google.adk.agents.callback_context import CallbackContext

async def test_gather_agent():
    class ListModel(RootModel[list[str]]):
        pass

    splitter = LlmAgent(
        name="pattern_continuist",
        model="gemini-2.0-flash",
        instruction = """
    Continue the input into a 3 item pattern.
    "5" -> ["5","6","7"]
    "42" -> ["42","43","44"]
    Always 3 items, always consecutive.
    """,
        output_schema=ListModel,
        output_key='list_key',
        disallow_transfer_to_parent=True,
        disallow_transfer_to_peers=True,
    )

    assert splitter.output_key

    gather = GatherAgent(
        name="TestGatherAgent",
        input_key="list",
        output_key="gather_output",
        sub_agents=[splitter],
        key_agent_name=splitter.name
    )

    expected_output = {tuple(str(j) for j in range(i, i + 3)) for i in range(20)}

    agent = AgentTester(
        initial_state={"list": [str(i) for i in range(20)]},
        agent=gather,
        check=lambda e: True,
        extract=lambda e: e.actions.state_delta
    )
    async for state_delta in agent.run(""):
        if gather.output_key in state_delta:
            gather_output = state_delta[gather.output_key]
        else:
            _, splitter_output = state_delta.popitem()
            assert isinstance(splitter_output, list)
            assert tuple(splitter_output) in expected_output
            

    assert agent.session is not None
    res = agent.session.state.get(gather.output_key)
    assert gather_output == res
    assert isinstance(res, list)
    assert set([tuple(x) for x in res]) == expected_output



def main() -> None:
    asyncio.run(test_gather_agent())


if __name__ == '__main__':
    main()