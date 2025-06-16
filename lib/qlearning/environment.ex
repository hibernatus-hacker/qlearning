defmodule QLearning.Environment do
  @moduledoc """
  Behaviour for defining reinforcement learning environments.

  An environment represents the world in which an agent operates. It defines
  the state space, action space, and the dynamics of how actions affect states
  and generate rewards.
  """

  @type state :: any()
  @type action :: any()
  @type reward :: number()

  @doc """
  Returns the list of all possible states in the environment.
  """
  @callback get_states() :: [state()]

  @doc """
  Returns the list of all possible actions in the environment.
  """
  @callback get_actions() :: [action()]

  @doc """
  Resets the environment to an initial state and returns that state.
  """
  @callback reset() :: state()

  @doc """
  Executes an action in the current state and returns:
  - next_state: The state after taking the action
  - reward: The immediate reward for taking this action
  - done: Boolean indicating if the episode is complete
  """
  @callback step(state(), action()) :: {state(), reward(), boolean()}

  @doc """
  Returns the list of valid actions from a given state.
  Default implementation returns all actions.
  """
  @callback get_valid_actions(state()) :: [action()]

  @optional_callbacks get_valid_actions: 1

  def get_states(module), do: module.get_states()
  def get_actions(module), do: module.get_actions()
  def reset(module), do: module.reset()
  def step(module, state, action), do: module.step(state, action)
  
  def get_valid_actions(module, state) do
    if function_exported?(module, :get_valid_actions, 1) do
      module.get_valid_actions(state)
    else
      module.get_actions()
    end
  end
end