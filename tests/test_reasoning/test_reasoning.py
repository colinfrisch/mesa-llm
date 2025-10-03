from unittest.mock import Mock

from mesa_llm.reasoning.reasoning import (
    Observation,
    Plan,
    Reasoning,
)


class TestObservation:
    """Test the Observation dataclass."""

    def test_observation_creation(self):
        """Test creating an Observation with valid data."""
        obs = Observation(
            step=1,
            self_state={"position": (0, 0), "health": 100},
            local_state={"Agent_1": {"position": (1, 1), "health": 90}},
        )

        assert obs.step == 1
        assert obs.self_state["position"] == (0, 0)
        assert obs.self_state["health"] == 100
        assert "Agent_1" in obs.local_state
        assert obs.local_state["Agent_1"]["position"] == (1, 1)


class TestPlan:
    """Test the Plan dataclass."""

    def test_plan_creation(self):
        """Test creating a Plan with valid data."""
        mock_llm_response = Mock()
        mock_llm_response.content = "Test plan content"

        plan = Plan(step=1, llm_plan=mock_llm_response, ttl=3)

        assert plan.step == 1
        assert plan.llm_plan == mock_llm_response
        assert plan.ttl == 3


class TestReasoningBase:
    """Tests for the abstract Reasoning base class."""

    def test_execute_tool_call_generates_plan(self):
        """Test that the base execute_tool_call method produces a Plan."""
        # 1. Setup a mock agent with all necessary components
        mock_agent = Mock()
        mock_agent.model.steps = 5

        # Mock the LLM and its response
        mock_llm_response = Mock()
        mock_llm_response.choices = [Mock()]
        mock_llm_response.choices[0].message = "Final LLM message"
        mock_agent.llm.generate.return_value = mock_llm_response

        # Mock the Tool Manager
        mock_agent.tool_manager.get_all_tools_schema.return_value = [
            {"schema": "example"}
        ]

        # 2. Instantiate a concrete implementation of Reasoning to test the base method
        class ConcreteReasoning(Reasoning):
            def plan(self, prompt, obs=None, ttl=1, selected_tools=None):
                pass  # Not needed for this test

        reasoning = ConcreteReasoning(agent=mock_agent)

        # 3. Call the method we want to test
        chaining_message = "Execute the plan."
        result_plan = reasoning.execute_tool_call(
            chaining_message, selected_tools=["tool1"]
        )

        # 4. Assert the results
        # Assert that the LLM was called with the correct parameters
        mock_agent.llm.generate.assert_called_once_with(
            prompt=chaining_message,
            tool_schema=[{"schema": "example"}],
            tool_choice="required",
        )
        # Assert that the tool manager was asked for the correct schema
        mock_agent.tool_manager.get_all_tools_schema.assert_called_once_with(
            selected_tools=["tool1"]
        )
        # Assert that the output is a correctly formed Plan object
        assert isinstance(result_plan, Plan)
        assert result_plan.step == 5
        assert result_plan.llm_plan == "Final LLM message"
