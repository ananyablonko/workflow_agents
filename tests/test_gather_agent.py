import __init__
import asyncio
import dotenv
import pytest

dotenv.load_dotenv()
import logging, sys
logger = logging.Logger(__name__)
logging.basicConfig(level=logging.INFO, stream=sys.stdout)
from pydantic import RootModel
from google.adk.agents import LlmAgent
from gather_agent import GatherAgent
from google.adk.agents.callback_context import CallbackContext

from ..utils.adk.app import AdkApp
from ..utils.adk.tester import create_test_session
from ..utils.text.printing import shorten

async def display_state(callback_context: CallbackContext) -> None:
    logger.debug(
    f"""user_content={shorten(callback_context.user_content, 150)}

invocation_id={shorten(callback_context.invocation_id, 150)}

agent_name={shorten(callback_context.agent_name, 150)}

state={shorten(callback_context.state.to_dict(), 150)}

""")

@pytest.mark.asyncio
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
        after_agent_callback=display_state
    )

    assert splitter.output_key

    gather = GatherAgent(
        name="TestGatherAgent",
        input_key="list",
        output_key="gather_output",
        sub_agents=[splitter],
        key_agent_name=splitter.name
    )
    n_runs = 4

    expected_output = {tuple(str(j) for j in range(i, i + 3)) for i in range(n_runs)}

    session = await create_test_session(AdkApp(
        name="TestGather",
        agent=gather,
        initial_state={"list": [str(i) for i in range(n_runs)]},
        check=lambda e: True,
        extract=lambda e: e.actions.state_delta
    ))

    async for state_delta in session.run(""):
        if gather.output_key in state_delta:
            gather_output = state_delta[gather.output_key]
        else:
            _, splitter_output = state_delta.popitem()
            assert isinstance(splitter_output, list)
            assert tuple(splitter_output) in expected_output
            

    res = session.state.get(gather.output_key)
    assert gather_output == res
    assert isinstance(res, list)
    assert set([tuple(x) for x in res]) == expected_output



def main() -> None:
    asyncio.run(test_gather_agent())


if __name__ == '__main__':
    main()