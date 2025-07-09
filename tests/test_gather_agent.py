import __init__
import asyncio
import pytest
import json
import random
from itertools import chain

from google.adk.agents import SequentialAgent, LoopAgent, ParallelAgent
from google.adk.events import Event, EventActions
from google.adk.agents.invocation_context import InvocationContext
from google.adk.agents.callback_context import CallbackContext
from google.genai import types

from gather_agent import GatherAgent
from lambda_agent import LambdaAgent
from utils.adk.app import AdkApp
from utils.adk.tester import create_test_session, MockAgent

def mock_response(ctx: InvocationContext):
    agent_input = ((ctx.user_content or types.Content()).parts or [types.Part()])[-1].text
    assert agent_input is not None
    assert agent_input.isnumeric()
    return json.dumps([str(int(agent_input) + i) for i in range(3)])


@pytest.mark.asyncio
async def test_gather_agent_text_input():
    gather = GatherAgent(
        name="gather_agent",
        output_key="gather_output",
        sub_agents=[
            MockAgent(
                mock_response=mock_response,
            )
        ],
    )

    session = await create_test_session(AdkApp(
        name="TestRegularInput",
        agent=gather,
        check=lambda _: True,
        extract=lambda e: e.actions.state_delta
    ))

    n_runs = 100

    input_data = json.dumps([str(i) for i in range(n_runs)])
    expected_output = {json.dumps([str(j) for j in range(i, i + 3)]) for i in range(n_runs)}

    async for state_delta in session.run(input_data):
        if gather.output_key in state_delta:
            gather_output = state_delta[gather.output_key]
        else:
            subagent_output = next((v for v in iter(state_delta.values())), None)
            assert isinstance(subagent_output, str)
            assert subagent_output in expected_output
            

    res = session.state.get(gather.output_key or "", {})
    assert gather_output == res
    assert "mock_agent" in res
    res_value = res["mock_agent"]
    assert isinstance(res_value, list)
    assert [json.loads(x)[0] == i for i, x in enumerate(res_value)]
    assert set(res_value) == expected_output


@pytest.mark.asyncio
async def test_gather_agent_with_input_key():
    mock_agent = MockAgent(
        name="input_key_mock",
        mock_response=mock_response,
    )
    
    gather = GatherAgent(
        name="gather_agent",
        output_key="gather_output",
        input_key="test_input_key",
        sub_agents=[mock_agent],
    )
    
    input_data = ["0", "1"]
    expected_output = {json.dumps([str(j) for j in range(i, i + 3)]) for i in range(len(input_data))}
    
    session = await create_test_session(AdkApp(
        name="TestInputKey",
        agent=gather,
        initial_state={"test_input_key": input_data},
        check=lambda _: True,
        extract=lambda e: e.actions.state_delta
    ))
    
    async for state_delta in session.run(""):
        if gather.output_key in state_delta:
            gather_output = state_delta[gather.output_key]
        else:
            subagent_output = next((v for v in iter(state_delta.values())), None)
            assert isinstance(subagent_output, str)
            assert subagent_output in expected_output
    
    res = session.state.get(gather.output_key or "", {})
    assert gather_output == res
    assert "input_key_mock" in res
    res_value = res["input_key_mock"]
    assert isinstance(res_value, list)
    assert set(res_value) == expected_output


@pytest.mark.asyncio
async def test_gather_agent_with_loop_agent():
    mock_leaf = MockAgent(
        name="loop_leaf",
        mock_response=mock_response,
    )
    
    async def lambda_func(ctx: InvocationContext):
        return Event(
            author=ctx.agent.name,
            actions=EventActions(escalate=random.random() > 0.5),
        )
    
    lambda_agent = LambdaAgent(
        name="loop_lambda",
        func=lambda_func
    )
    
    loop_agent = LoopAgent(
        name="test_loop",
        sub_agents=[mock_leaf, lambda_agent],
        max_iterations=2,
    )
    
    gather = GatherAgent(
        name="gather_agent",
        output_key="gather_output",
        sub_agents=[loop_agent],
    )
    
    session = await create_test_session(AdkApp(
        name="TestLoopAgent",
        agent=gather,
        check=lambda _: True,
        extract=lambda e: e.actions.state_delta
    ))
    
    input_data = json.dumps(["0", "1"])
    expected_output = {json.dumps([str(j) for j in range(i, i + 3)]) for i in range(2)}
    
    async for state_delta in session.run(input_data):
        if gather.output_key in state_delta:
            gather_output = state_delta[gather.output_key]
        else:
            subagent_output = next((v for v in iter(state_delta.values())), None)
            if subagent_output:
                assert subagent_output in expected_output
    
    res = session.state.get(gather.output_key or "", {})
    assert gather_output == res
    assert "loop_leaf" in res
    res_value = res["loop_leaf"]
    assert isinstance(res_value, list)
    assert set(res_value) == expected_output


@pytest.mark.asyncio
async def test_gather_agent_with_parallel_agent():
    mock1 = MockAgent(
        name="parallel_mock1",
        mock_response=mock_response,
    )
    
    mock2 = MockAgent(
        name="parallel_mock2", 
        mock_response=mock_response,
    )
    
    parallel_agent = ParallelAgent(
        name="test_parallel",
        sub_agents=[mock1, mock2],
    )
    
    gather = GatherAgent(
        name="gather_agent",
        output_key="gather_output",
        sub_agents=[parallel_agent],
    )
    
    session = await create_test_session(AdkApp(
        name="TestParallelAgent",
        agent=gather,
        check=lambda _: True,
        extract=lambda e: e.actions.state_delta
    ))
    
    input_data = json.dumps(["0", "1"])
    expected_output = {json.dumps([str(j) for j in range(i, i + 3)]) for i in range(2)}
    
    async for state_delta in session.run(input_data):
        if gather.output_key in state_delta:
            gather_output = state_delta[gather.output_key]
        else:
            subagent_output = next((v for v in iter(state_delta.values())), None)
            if subagent_output:
                assert subagent_output in expected_output
    
    res = session.state.get(gather.output_key or "", {})
    assert gather_output == res
    assert "parallel_mock1" in res
    assert "parallel_mock2" in res
    res_value1 = res["parallel_mock1"]
    res_value2 = res["parallel_mock2"]
    assert isinstance(res_value1, list)
    assert isinstance(res_value2, list)
    assert set(res_value1) == expected_output
    assert set(res_value2) == expected_output


@pytest.mark.asyncio
async def test_gather_agent_with_sequential_agent():
    mock1 = MockAgent(
        name="seq_mock1",
        mock_response=mock_response,
    )
    
    mock2 = MockAgent(
        name="seq_mock2",
        mock_response=mock_response,
    )
    
    sequential_agent = SequentialAgent(
        name="test_sequential",
        sub_agents=[mock1, mock2],
    )
    
    gather = GatherAgent(
        name="gather_agent",
        output_key="gather_output",
        sub_agents=[sequential_agent],
    )
    
    session = await create_test_session(AdkApp(
        name="TestSequentialAgent",
        agent=gather,
        check=lambda _: True,
        extract=lambda e: e.actions.state_delta
    ))
    
    input_data = json.dumps(["0", "1"])
    expected_output = {json.dumps([str(j) for j in range(i, i + 3)]) for i in range(2)}
    
    async for state_delta in session.run(input_data):
        if gather.output_key in state_delta:
            gather_output = state_delta[gather.output_key]
        else:
            subagent_output = next((v for v in iter(state_delta.values())), None)
            if subagent_output and isinstance(subagent_output, str):
                assert subagent_output in expected_output
    
    res = session.state.get(gather.output_key or "", {})
    assert gather_output == res
    assert "seq_mock1" in res
    assert "seq_mock2" in res
    res_value1 = res["seq_mock1"]
    res_value2 = res["seq_mock2"]
    assert isinstance(res_value1, list)
    assert isinstance(res_value2, list)
    all_outputs = set(res_value1 + res_value2)
    assert all_outputs == expected_output


@pytest.mark.asyncio
async def test_gather_agent_with_gather_agent():
    mock_leaf = MockAgent(
        name="nested_mock",
        mock_response=mock_response,
    )
    
    inner_gather = GatherAgent(
        name="inner_gather",
        sub_agents=[mock_leaf],
    )
    
    outer_gather = GatherAgent(
        name="outer_gather", 
        output_key="outer_gather_output",
        sub_agents=[inner_gather],
    )
    
    session = await create_test_session(AdkApp(
        name="TestNestedGather",
        agent=outer_gather,
        check=lambda _: True,
        extract=lambda e: e.actions.state_delta
    ))
    
    input_data = json.dumps([json.dumps([str(i), str(i + 1)]) for i in [10, 20, 30]])
    expected_output = {json.dumps([str(j) for j in range(i, i + 3)]) for i in [10, 11, 20, 21, 30, 31]}
    
    async for state_delta in session.run(input_data):
        if outer_gather.output_key in state_delta:
            gather_output = state_delta[outer_gather.output_key]
        else:
            subagent_output = next((v for v in iter(state_delta.values())), None)
            if subagent_output and isinstance(subagent_output, str):
                assert subagent_output in expected_output
    
    res = session.state.get(outer_gather.output_key or "", {})
    assert gather_output == res
    assert "inner_gather" in res
    res_value = res["inner_gather"]
    assert isinstance(res_value, list)
    assert len(res_value) == 3
    assert all(len(list(x.values())[0]) == 2 for x in res_value)
    assert set(sum((sum((v for v in d.values()), []) for d in res_value), [])) == expected_output


@pytest.mark.asyncio
async def test_gather_agent_tree():
    def seq_agent(name: str):
        return SequentialAgent(
            name=name,
            sub_agents=[
                MockAgent(name=f"{name}_leaf_1", mock_response=mock_response),
                MockAgent(name=f"{name}_leaf_2", mock_response=mock_response),
            ],
        )

    inner_gather = GatherAgent(
        name="inner_gather",
        sub_agents=[seq_agent("seq_inner")],
        input_key="inner_gather_input",
    )

    main_parallel = ParallelAgent(
        name="main_parallel",
        sub_agents=[seq_agent("seq_branch"), inner_gather],
    )

    main_gather = GatherAgent(
        name="outer_gather",
        output_key="deep_tree_output",
        sub_agents=[main_parallel],
    )

    session = await create_test_session(AdkApp(
        name="TestDeepTree",
        agent=main_gather,
        initial_state={
            "inner_gather_input": ["0", "1"]
        },
        check=lambda _: True,
        extract=lambda e: e.actions.state_delta
    ))

    input_data = json.dumps(["0", "1"])
    expected_output = {json.dumps([str(j) for j in range(i, i + 3)]) for i in range(2)}

    async for state_delta in session.run(input_data):
        if main_gather.output_key in state_delta:
            gather_output = state_delta[main_gather.output_key]
        else:
            subagent_output = next((v for v in iter(state_delta.values())), None)
            if subagent_output and isinstance(subagent_output, str):
                assert subagent_output in expected_output

    res = session.state.get(main_gather.output_key or "", {})
    assert gather_output == res
    assert "seq_branch_leaf_1" in res
    assert "seq_branch_leaf_2" in res
    assert "inner_gather" in res
    res_seq1 = res["seq_branch_leaf_1"]
    res_seq2 = res["seq_branch_leaf_2"]
    res_inner = res["inner_gather"]
    assert isinstance(res_seq1, list)
    assert isinstance(res_seq2, list)
    assert isinstance(res_inner, list)
    all_leaf_outputs = set(res_seq1 + res_seq2)
    assert all_leaf_outputs == expected_output


@pytest.mark.asyncio
async def test_recursive_agent():
    def seven_boom(callback_context: CallbackContext) -> None:
        agent_output = callback_context.state['gather_output']
        new_input = chain(*[json.loads(x) for x in agent_output['mock_agent']])
        filtered = [x for x in new_input if '7' not in x and int(x) % 7 != 0]
        callback_context.state['numbers'] = filtered

    gather_agent = GatherAgent(
        name='gather',
        sub_agents=[MockAgent(mock_response=mock_response)],
        input_key='numbers',
        output_key='gather_output',
        after_agent_callback=seven_boom,
    )

    recursive_agent = LoopAgent(
        name='recursive_agent',
        sub_agents=[gather_agent],
        max_iterations=4
    )

    session = await create_test_session(AdkApp(
        agent=recursive_agent,
        name="test_recursive_agent",
        initial_state={'numbers': ['3']},
        check=lambda _: True,
        extract=lambda e: e.actions.state_delta,
    ))

    expected_output = {json.dumps([str(j) for j in range(i, i + 3)]) for i in [3, 4, 5, 6, 8, 9, 10]}

    async for state_delta in session.run(""):
        if 'numbers' not in state_delta and gather_agent.output_key not in state_delta:
            subagent_output = next((v for v in iter(state_delta.values())), None)
            if subagent_output and isinstance(subagent_output, str):
                assert subagent_output in expected_output


if __name__ == '__main__':
    async def run_all_tests():
        await test_gather_agent_text_input()
        await test_gather_agent_with_input_key()
        await test_gather_agent_with_loop_agent()
        await test_gather_agent_with_parallel_agent()
        await test_gather_agent_with_sequential_agent()
        await test_gather_agent_with_gather_agent()
        await test_gather_agent_tree()
        await test_recursive_agent()
        
    asyncio.run(run_all_tests())
