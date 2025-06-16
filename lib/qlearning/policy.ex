defmodule QLearning.Policy do
  @moduledoc """
  Policy modules for action selection in reinforcement learning.
  
  Provides various exploration strategies including epsilon-greedy,
  Upper Confidence Bound (UCB), and Boltzmann exploration.
  """

  @type policy :: :epsilon_greedy | :ucb | :boltzmann | :greedy
  @type q_values :: Nx.Tensor.t() | map()
  @type action :: integer() | any()

  @doc """
  Selects an action based on the specified policy.
  
  ## Parameters
  - `policy`: The policy type to use
  - `q_values`: Q-values for available actions (tensor or map)
  - `options`: Policy-specific options
  """
  def select_action(policy, q_values, options \\ %{})

  def select_action(:epsilon_greedy, q_values, options) do
    epsilon = Map.get(options, :epsilon, 0.1)
    __MODULE__.EpsilonGreedy.select_action(q_values, epsilon)
  end

  def select_action(:ucb, q_values, options) do
    action_counts = Map.get(options, :action_counts, %{})
    total_steps = Map.get(options, :total_steps, 1)
    c = Map.get(options, :c, 2.0)
    __MODULE__.UpperConfidenceBound.select_action(q_values, action_counts, total_steps, c)
  end

  def select_action(:boltzmann, q_values, options) do
    temperature = Map.get(options, :temperature, 1.0)
    __MODULE__.Boltzmann.select_action(q_values, temperature)
  end

  def select_action(:greedy, q_values, _options) do
    __MODULE__.Greedy.select_action(q_values)
  end

  @doc """
  Creates an epsilon schedule that decays over time.
  """
  def epsilon_schedule(step, schedule_type \\ :linear, options \\ %{})

  def epsilon_schedule(step, :linear, options) do
    start_epsilon = Map.get(options, :start_epsilon, 1.0)
    end_epsilon = Map.get(options, :end_epsilon, 0.01)
    decay_steps = Map.get(options, :decay_steps, 1000)
    
    decay_rate = (start_epsilon - end_epsilon) / decay_steps
    epsilon = start_epsilon - decay_rate * step
    max(epsilon, end_epsilon)
  end

  def epsilon_schedule(step, :exponential, options) do
    start_epsilon = Map.get(options, :start_epsilon, 1.0)
    end_epsilon = Map.get(options, :end_epsilon, 0.01)
    decay_rate = Map.get(options, :decay_rate, 0.995)
    
    epsilon = start_epsilon * :math.pow(decay_rate, step)
    max(epsilon, end_epsilon)
  end

  def epsilon_schedule(step, :cosine, options) do
    start_epsilon = Map.get(options, :start_epsilon, 1.0)
    end_epsilon = Map.get(options, :end_epsilon, 0.01)
    decay_steps = Map.get(options, :decay_steps, 1000)
    
    progress = min(step / decay_steps, 1.0)
    cosine_decay = 0.5 * (1 + :math.cos(:math.pi() * progress))
    epsilon = end_epsilon + (start_epsilon - end_epsilon) * cosine_decay
    epsilon
  end

  # Individual policy implementations

  defmodule EpsilonGreedy do
    @moduledoc """
    Epsilon-greedy policy: explores with probability epsilon, exploits otherwise.
    """

    def select_action(%Nx.Tensor{} = q_values, epsilon) do
      num_actions = Nx.axis_size(q_values, -1)
      
      if :rand.uniform() < epsilon do
        # Exploration: random action
        :rand.uniform(num_actions) - 1
      else
        # Exploitation: best action
        q_values
        |> Nx.argmax(axis: -1)
        |> Nx.to_number()
      end
    end

    def select_action(q_values, epsilon) when is_map(q_values) do
      if :rand.uniform() < epsilon do
        # Exploration: random action
        q_values |> Map.keys() |> Enum.random()
      else
        # Exploitation: best action
        q_values
        |> Enum.max_by(fn {_action, value} -> value end)
        |> elem(0)
      end
    end
  end

  defmodule Greedy do
    @moduledoc """
    Greedy policy: always selects the action with highest Q-value.
    """

    def select_action(%Nx.Tensor{} = q_values) do
      q_values
      |> Nx.argmax(axis: -1)
      |> Nx.to_number()
    end

    def select_action(q_values) when is_map(q_values) do
      q_values
      |> Enum.max_by(fn {_action, value} -> value end)
      |> elem(0)
    end
  end

  defmodule UpperConfidenceBound do
    @moduledoc """
    Upper Confidence Bound (UCB) policy for multi-armed bandit problems.
    
    Balances exploration and exploitation by considering both Q-values
    and action uncertainty.
    """

    def select_action(q_values, action_counts, total_steps, c \\ 2.0)
    
    def select_action(%Nx.Tensor{} = q_values, action_counts, total_steps, c) do
      import Nx.Defn
      
      num_actions = Nx.axis_size(q_values, -1)
      
      # Convert action counts to tensor
      counts_list = 
        0..(num_actions - 1)
        |> Enum.map(fn i -> Map.get(action_counts, i, 0) end)
      
      counts = Nx.tensor(counts_list, type: :f32)
      
      # Calculate UCB values
      log_total = :math.log(max(total_steps, 1))
      
      # Avoid division by zero by using a small epsilon for zero counts
      safe_counts = Nx.select(
        Nx.greater(counts, 0),
        counts,
        Nx.broadcast(1.0e-10, Nx.shape(counts))
      )
      
      confidence = Nx.select(
        Nx.greater(counts, 0),
        Nx.sqrt(Nx.divide(log_total, safe_counts)),
        Nx.broadcast(1.0e6, Nx.shape(counts))  # Large value for unvisited actions
      )
      
      ucb_values = Nx.add(q_values, Nx.multiply(c, confidence))
      
      ucb_values
      |> Nx.argmax(axis: -1)
      |> Nx.to_number()
    end
    
    def select_action(q_values, action_counts, total_steps, c) when is_map(q_values) do
      ucb_values = 
        q_values
        |> Enum.map(fn {action, q_value} ->
          count = Map.get(action_counts, action, 0)
          
          confidence = if count > 0 do
            c * :math.sqrt(:math.log(total_steps) / count)
          else
            Float.max_finite()  # Unvisited actions get infinite confidence
          end
          
          {action, q_value + confidence}
        end)
        |> Enum.into(%{})
      
      ucb_values
      |> Enum.max_by(fn {_action, ucb_value} -> ucb_value end)
      |> elem(0)
    end
  end

  defmodule Boltzmann do
    @moduledoc """
    Boltzmann (softmax) exploration policy.
    
    Selects actions probabilistically based on their Q-values,
    with temperature controlling exploration vs exploitation.
    """

    def select_action(q_values, temperature \\ 1.0)
    
    def select_action(%Nx.Tensor{} = q_values, temperature) do
      # Apply softmax with temperature
      scaled_q = Nx.divide(q_values, temperature)
      # Manual softmax implementation since Nx.softmax may not be available
      exp_vals = Nx.exp(Nx.subtract(scaled_q, Nx.reduce_max(scaled_q)))
      probabilities = Nx.divide(exp_vals, Nx.sum(exp_vals))
      
      # Sample from categorical distribution
      sample_categorical(probabilities)
    end
    
    def select_action(q_values, temperature) when is_map(q_values) do
      # Convert to list of {action, q_value} pairs
      q_list = Enum.to_list(q_values)
      
      # Calculate probabilities using softmax
      max_q = q_list |> Enum.map(fn {_, q} -> q end) |> Enum.max()
      
      exp_values = 
        q_list
        |> Enum.map(fn {action, q} -> 
          {action, :math.exp((q - max_q) / temperature)}
        end)
      
      total_exp = exp_values |> Enum.map(fn {_, exp_val} -> exp_val end) |> Enum.sum()
      
      probabilities = 
        exp_values
        |> Enum.map(fn {action, exp_val} -> {action, exp_val / total_exp} end)
      
      # Sample from the probability distribution
      sample_action(probabilities)
    end

    defp sample_action(probabilities) do
      rand = :rand.uniform()
      
      probabilities
      |> Enum.reduce_while({0.0, nil}, fn {action, prob}, {cumulative, _} ->
        new_cumulative = cumulative + prob
        if rand <= new_cumulative do
          {:halt, {new_cumulative, action}}
        else
          {:cont, {new_cumulative, action}}
        end
      end)
      |> elem(1)
    end

    defp sample_categorical(probabilities) do
      # Convert to Elixir and sample
      probs_list = Nx.to_list(probabilities)
      rand = :rand.uniform()
      
      probs_list
      |> Enum.with_index()
      |> Enum.reduce_while({0.0, 0}, fn {prob, idx}, {cumulative, _} ->
        new_cumulative = cumulative + prob
        if rand <= new_cumulative do
          {:halt, {new_cumulative, idx}}
        else
          {:cont, {new_cumulative, idx}}
        end
      end)
      |> elem(1)
    end
  end

  # Policy evaluation and comparison utilities

  @doc """
  Evaluates a policy by running episodes and returning average reward.
  """
  def evaluate_policy(environment, policy, q_table, episodes \\ 100, max_steps \\ 1000) do
    total_rewards = 
      1..episodes
      |> Enum.map(fn _episode ->
        evaluate_single_episode(environment, policy, q_table, max_steps)
      end)
    
    Enum.sum(total_rewards) / episodes
  end

  defp evaluate_single_episode(environment, policy, q_table, max_steps) do
    state = QLearning.Environment.reset(environment)
    actions = QLearning.Environment.get_actions(environment)
    
    {_final_state, total_reward} = 
      1..max_steps
      |> Enum.reduce_while({state, 0.0}, fn _step, {current_state, acc_reward} ->
        # Get Q-values for current state
        state_q_values = 
          actions
          |> Enum.map(fn action -> {action, Map.get(q_table, {current_state, action}, 0.0)} end)
          |> Enum.into(%{})
        
        # Select action using policy (no exploration for evaluation)
        action = case policy do
          :epsilon_greedy -> select_action(:greedy, state_q_values)
          _ -> select_action(policy, state_q_values)
        end
        
        {next_state, reward, done} = QLearning.Environment.step(environment, current_state, action)
        
        if done do
          {:halt, {next_state, acc_reward + reward}}
        else
          {:cont, {next_state, acc_reward + reward}}
        end
      end)
    
    total_reward
  end

  @doc """
  Adaptive epsilon that adjusts based on performance.
  """
  def adaptive_epsilon(current_epsilon, recent_rewards, target_reward, options \\ %{}) do
    min_epsilon = Map.get(options, :min_epsilon, 0.01)
    max_epsilon = Map.get(options, :max_epsilon, 1.0)
    adaptation_rate = Map.get(options, :adaptation_rate, 0.1)
    
    avg_reward = Enum.sum(recent_rewards) / max(length(recent_rewards), 1)
    
    if avg_reward < target_reward do
      # Increase exploration if performance is poor
      new_epsilon = current_epsilon + adaptation_rate * (max_epsilon - current_epsilon)
      min(new_epsilon, max_epsilon)
    else
      # Decrease exploration if performance is good
      new_epsilon = current_epsilon - adaptation_rate * (current_epsilon - min_epsilon)
      max(new_epsilon, min_epsilon)
    end
  end
end