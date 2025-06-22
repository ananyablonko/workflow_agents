import __init__

import asyncio

from google.adk.agents import LlmAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from verbose_agent import VerboseAgent

# import logging
# logging.basicConfig(level=logging.INFO)


async def test_verbose_agent():
    agent = VerboseAgent(LlmAgent(
            name='greeter',
            description="Greets the user",
            model="gemini-2.0-flash",
        ),
        before_agent_message="Before",
        after_agent_message="After",
    )

    runner = Runner(
        app_name="TestVerboseAgent",
        agent=agent,
        session_service=InMemorySessionService(),
    )

    await runner.session_service.create_session(app_name=runner.app_name, user_id="0", session_id="0")
    message = types.Content(role="user", parts=[types.Part(text="Hi")])
    responses = []
    async for event in runner.run_async(user_id="0", session_id="0", new_message=message):
        if event.is_final_response() and event.content and event.content.parts:
            print(event.content.parts[0].text)
            responses.append(event.content.parts[0].text)
    
    assert len(responses) == 3
    assert responses[0] == "Before"
    assert responses[2] == "After"



async def main() -> None:
    await test_verbose_agent()


if __name__ == '__main__':
    asyncio.run(main())