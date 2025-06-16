defmodule QLearning.EnvironmentTest do
  use ExUnit.Case
  alias QLearning.Environment

  defmodule TestEnvironment do
    @behaviour QLearning.Environment

    @impl true
    def get_states(), do: [:state1, :state2, :state3]

    @impl true 
    def get_actions(), do: [:action1, :action2]

    @impl true
    def reset(), do: :state1

    @impl true
    def step(:state1, :action1), do: {:state2, 1.0, false}
    def step(:state1, :action2), do: {:state3, -1.0, false}
    def step(:state2, :action1), do: {:state3, 5.0, true}
    def step(:state2, :action2), do: {:state1, 0.0, false}
    def step(:state3, _action), do: {:state3, 0.0, true}

    @impl true
    def get_valid_actions(:state1), do: [:action1, :action2]
    def get_valid_actions(:state2), do: [:action1, :action2]
    def get_valid_actions(:state3), do: []
  end

  defmodule MinimalEnvironment do
    @behaviour QLearning.Environment

    @impl true
    def get_states(), do: [:single_state]

    @impl true
    def get_actions(), do: [:single_action]

    @impl true
    def reset(), do: :single_state

    @impl true
    def step(_state, _action), do: {:single_state, 0.0, true}
  end

  describe "Environment behaviour wrapper functions" do
    test "get_states/1 delegates to module" do
      states = Environment.get_states(TestEnvironment)
      assert states == [:state1, :state2, :state3]
    end

    test "get_actions/1 delegates to module" do
      actions = Environment.get_actions(TestEnvironment)
      assert actions == [:action1, :action2]
    end

    test "reset/1 delegates to module" do
      initial_state = Environment.reset(TestEnvironment)
      assert initial_state == :state1
    end

    test "step/3 delegates to module" do
      {next_state, reward, done} = Environment.step(TestEnvironment, :state1, :action1)
      assert next_state == :state2
      assert reward == 1.0
      assert done == false
    end

    test "get_valid_actions/2 delegates to module when implemented" do
      valid_actions = Environment.get_valid_actions(TestEnvironment, :state1)
      assert valid_actions == [:action1, :action2]
    end

    test "get_valid_actions/2 falls back to get_actions when not implemented" do
      valid_actions = Environment.get_valid_actions(MinimalEnvironment, :single_state)
      assert valid_actions == [:single_action]
    end
  end

  describe "TestEnvironment transitions" do
    test "transitions from state1 work correctly" do
      {next_state, reward, done} = Environment.step(TestEnvironment, :state1, :action1)
      assert {next_state, reward, done} == {:state2, 1.0, false}

      {next_state, reward, done} = Environment.step(TestEnvironment, :state1, :action2)
      assert {next_state, reward, done} == {:state3, -1.0, false}
    end

    test "transitions from state2 work correctly" do
      {next_state, reward, done} = Environment.step(TestEnvironment, :state2, :action1)
      assert {next_state, reward, done} == {:state3, 5.0, true}

      {next_state, reward, done} = Environment.step(TestEnvironment, :state2, :action2)
      assert {next_state, reward, done} == {:state1, 0.0, false}
    end

    test "terminal state3 stays terminal" do
      {next_state, reward, done} = Environment.step(TestEnvironment, :state3, :action1)
      assert {next_state, reward, done} == {:state3, 0.0, true}

      {next_state, reward, done} = Environment.step(TestEnvironment, :state3, :action2)
      assert {next_state, reward, done} == {:state3, 0.0, true}
    end
  end

  describe "valid actions" do
    test "returns available actions for non-terminal states" do
      assert Environment.get_valid_actions(TestEnvironment, :state1) == [:action1, :action2]
      assert Environment.get_valid_actions(TestEnvironment, :state2) == [:action1, :action2]
    end

    test "returns empty list for terminal state" do
      assert Environment.get_valid_actions(TestEnvironment, :state3) == []
    end
  end
end