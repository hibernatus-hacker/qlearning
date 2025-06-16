defmodule QLearning do
  @moduledoc """
  A comprehensive Q-learning reinforcement learning library for Elixir.

  Q-learning is a model-free reinforcement learning algorithm that learns
  the value of actions in particular states. It does not require a model
  of the environment and can handle stochastic transitions and rewards.

  ## Basic Usage

      iex> states = [:s1, :s2, :s3]
      iex> actions = [:up, :down, :left, :right]
      iex> q_table = QLearning.init_q_table(states, actions)
      iex> action = QLearning.choose_action(:s1, q_table, actions, 0.1)
      iex> new_q_table = QLearning.update_q_value(q_table, :s1, action, 1.0, :s2, actions)

  ## Parameters

  - **Alpha (Learning Rate)**: Controls how much new information overrides old information (0.0 to 1.0)
  - **Gamma (Discount Factor)**: Determines importance of future rewards (0.0 to 1.0)
  - **Epsilon**: Exploration rate for epsilon-greedy policy (0.0 to 1.0)
  """

  alias QLearning.{Agent, Environment, Policy}

  @type state :: any()
  @type action :: any()
  @type reward :: number()
  @type q_table :: %{{state(), action()} => float()}
  @type hyperparams :: %{
    alpha: float(),
    gamma: float(),
    epsilon: float()
  }

  @doc """
  Initializes a Q-table with all Q-values set to zero.

  ## Parameters
  - `states`: List of all possible states
  - `actions`: List of all possible actions

  ## Examples

      iex> QLearning.init_q_table([:s1, :s2], [:up, :down])
      %{{:s1, :up} => 0.0, {:s1, :down} => 0.0, {:s2, :up} => 0.0, {:s2, :down} => 0.0}
  """
  @spec init_q_table([state()], [action()]) :: q_table()
  def init_q_table(states, actions) do
    for s <- states, a <- actions, into: %{} do
      {{s, a}, 0.0}
    end
  end

  @doc """
  Chooses an action using epsilon-greedy policy.

  With probability epsilon, chooses a random action (exploration).
  Otherwise, chooses the action with highest Q-value (exploitation).

  ## Parameters
  - `state`: Current state
  - `q_table`: Q-table mapping {state, action} pairs to values
  - `actions`: List of available actions
  - `epsilon`: Exploration rate (0.0 = pure exploitation, 1.0 = pure exploration)

  ## Examples

      iex> q_table = %{{:s1, :up} => 0.5, {:s1, :down} => 0.8}
      iex> QLearning.choose_action(:s1, q_table, [:up, :down], 0.0)
      :down
  """
  @spec choose_action(state(), q_table(), [action()], float()) :: action()
  def choose_action(state, q_table, actions, epsilon) do
    if :rand.uniform() < epsilon do
      # Exploration: random action
      Enum.random(actions)
    else
      # Exploitation: best action based on Q-values
      actions
      |> Enum.map(fn action -> {action, Map.get(q_table, {state, action}, 0.0)} end)
      |> Enum.max_by(fn {_action, value} -> value end)
      |> elem(0)
    end
  end

  @doc """
  Updates a single Q-value using the Q-learning update rule.

  Q(s,a) = Q(s,a) + α[r + γ max Q(s',a') - Q(s,a)]

  ## Parameters
  - `q_table`: Current Q-table
  - `state`: Current state
  - `action`: Action taken
  - `reward`: Reward received
  - `next_state`: Next state reached
  - `actions`: List of all possible actions
  - `alpha`: Learning rate (default: 0.1)
  - `gamma`: Discount factor (default: 0.9)

  ## Examples

      iex> q_table = %{{:s1, :up} => 0.0, {:s2, :up} => 0.5}
      iex> QLearning.update_q_value(q_table, :s1, :up, 1.0, :s2, [:up], 0.1, 0.9)
      %{{:s1, :up} => 0.145, {:s2, :up} => 0.5}
  """
  @spec update_q_value(q_table(), state(), action(), reward(), state(), [action()], float(), float()) :: q_table()
  def update_q_value(q_table, state, action, reward, next_state, actions, alpha \\ 0.1, gamma \\ 0.9) do
    current_q = Map.get(q_table, {state, action}, 0.0)
    
    # Find max Q-value for next state
    max_next_q = 
      actions
      |> Enum.map(fn a -> Map.get(q_table, {next_state, a}, 0.0) end)
      |> Enum.max(fn -> 0.0 end)
    
    # Q-learning update formula
    new_q = current_q + alpha * (reward + gamma * max_next_q - current_q)
    
    Map.put(q_table, {state, action}, new_q)
  end

  @doc """
  Trains a Q-learning agent for multiple episodes.

  ## Parameters
  - `environment`: Environment module implementing the Environment behaviour
  - `hyperparams`: Map containing alpha, gamma, and epsilon values
  - `episodes`: Number of episodes to train
  - `max_steps`: Maximum steps per episode (default: 1000)

  Returns the final Q-table after training.
  """
  @spec train(module(), hyperparams(), pos_integer(), pos_integer()) :: q_table()
  def train(environment, hyperparams, episodes, max_steps \\ 1000) do
    states = Environment.get_states(environment)
    actions = Environment.get_actions(environment)
    q_table = init_q_table(states, actions)
    
    1..episodes
    |> Enum.reduce(q_table, fn episode, acc_q_table ->
      train_episode(environment, acc_q_table, hyperparams, max_steps, episode)
    end)
  end

  @doc """
  Evaluates a trained Q-table by running episodes with pure exploitation (epsilon = 0).

  Returns average total reward over the evaluation episodes.
  """
  @spec evaluate(module(), q_table(), pos_integer(), pos_integer()) :: float()
  def evaluate(environment, q_table, episodes, max_steps \\ 1000) do
    actions = Environment.get_actions(environment)
    
    total_rewards = 
      1..episodes
      |> Enum.map(fn _episode ->
        evaluate_episode(environment, q_table, actions, max_steps)
      end)
    
    Enum.sum(total_rewards) / episodes
  end

  # Private functions

  defp train_episode(environment, q_table, %{alpha: alpha, gamma: gamma, epsilon: epsilon}, max_steps, episode) do
    state = Environment.reset(environment)
    actions = Environment.get_actions(environment)
    
    1..max_steps
    |> Enum.reduce_while(q_table, fn _step, acc_q_table ->
      action = choose_action(state, acc_q_table, actions, epsilon)
      {next_state, reward, done} = Environment.step(environment, state, action)
      
      new_q_table = update_q_value(acc_q_table, state, action, reward, next_state, actions, alpha, gamma)
      
      if done do
        {:halt, new_q_table}
      else
        {:cont, new_q_table}
      end
    end)
  end

  defp evaluate_episode(environment, q_table, actions, max_steps) do
    state = Environment.reset(environment)
    
    {_final_state, total_reward} = 
      1..max_steps
      |> Enum.reduce_while({state, 0.0}, fn _step, {current_state, acc_reward} ->
        action = choose_action(current_state, q_table, actions, 0.0) # No exploration
        {next_state, reward, done} = Environment.step(environment, current_state, action)
        
        if done do
          {:halt, {next_state, acc_reward + reward}}
        else
          {:cont, {next_state, acc_reward + reward}}
        end
      end)
    
    total_reward
  end
end