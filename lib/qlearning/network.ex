defmodule QLearning.Network do
  @moduledoc """
  Neural network utilities for Deep Q-learning using Axon.
  
  This module provides functions to build, initialize, and manage
  neural networks for approximating Q-value functions.
  """

  import Nx.Defn

  @doc """
  Creates a standard Deep Q-Network architecture.
  
  ## Parameters
  - `input_size`: Size of the state input
  - `output_size`: Number of possible actions
  - `hidden_sizes`: List of hidden layer sizes (default: [64, 64])
  - `activation`: Activation function (default: :relu)
  
  ## Examples
  
      iex> model = QLearning.Network.build_dqn(4, 2)
      iex> {init_fn, predict_fn} = Axon.build(model)
  """
  def build_dqn(input_size, output_size, hidden_sizes \\ [64, 64], activation \\ :relu) do
    Axon.input("state", shape: {nil, input_size})
    |> build_hidden_layers(hidden_sizes, activation)
    |> Axon.dense(output_size, name: "q_values")
  end

  @doc """
  Creates a dueling Deep Q-Network architecture.
  
  Dueling DQN separates the estimation of state values and action advantages,
  which can lead to better learning in some environments.
  """
  def build_dueling_dqn(input_size, output_size, hidden_sizes \\ [64, 64], activation \\ :relu) do
    input = Axon.input("state", shape: {nil, input_size})
    
    # Shared feature layers
    shared = build_hidden_layers(input, hidden_sizes, activation)
    
    # Value stream
    value_stream = 
      shared
      |> Axon.dense(32, activation: activation, name: "value_hidden")
      |> Axon.dense(1, name: "state_value")
    
    # Advantage stream
    advantage_stream = 
      shared
      |> Axon.dense(32, activation: activation, name: "advantage_hidden")
      |> Axon.dense(output_size, name: "advantages")
    
    # Combine streams: Q(s,a) = V(s) + A(s,a) - mean(A(s,:))
    Axon.layer(
      fn {value, advantages}, _opts ->
        mean_advantages = Nx.mean(advantages, axes: [-1], keep_axes: true)
        Nx.add(value, Nx.subtract(advantages, mean_advantages))
      end,
      [value_stream, advantage_stream],
      name: "dueling_q_values"
    )
  end

  @doc """
  Creates parameter sets for the networks.
  """
  def init_params(model, state_shape, key \\ Nx.Random.key(42)) do
    {init_fn, _predict_fn} = Axon.build(model)
    sample_input = %{"state" => Nx.Random.uniform(key, {1, elem(state_shape, 0)}, type: :f32)}
    init_fn.(sample_input, %{})
  end

  @doc """
  Predicts Q-values for given states.
  """
  def predict(model, params, states) do
    {_init_fn, predict_fn} = Axon.build(model)
    predict_fn.(params, %{"state" => states})
  end

  @doc """
  Computes the loss between predicted and target Q-values.
  Uses Huber loss for stability.
  """
  defn huber_loss(y_true, y_pred, delta \\ 1.0) do
    error = Nx.abs(Nx.subtract(y_true, y_pred))
    
    Nx.select(
      Nx.less_equal(error, delta),
      0.5 * Nx.pow(error, 2),
      delta * error - 0.5 * Nx.pow(delta, 2)
    )
    |> Nx.mean()
  end

  @doc """
  Computes Q-learning loss for a batch of experiences.
  """
  defn q_learning_loss(params, model_fn, states, actions, rewards, next_states, dones, gamma \\ 0.99) do
    # Current Q-values
    current_q_values = model_fn.(params, %{"state" => states})
    current_q = gather_q_values(current_q_values, actions)
    
    # Target Q-values (stop gradient to prevent training target network)
    next_q_values = Nx.Defn.stop_grad(model_fn.(params, %{"state" => next_states}))
    max_next_q = Nx.reduce_max(next_q_values, axes: [-1])
    
    # Target: r + γ * max(Q(s', a')) * (1 - done)
    targets = Nx.add(rewards, Nx.multiply(gamma, Nx.multiply(max_next_q, Nx.subtract(1.0, dones))))
    
    huber_loss(targets, current_q)
  end

  @doc """
  Updates network parameters using gradient descent.
  """
  def update_params(params, gradients, learning_rate \\ 0.001) do
    Polaris.Updates.scale_by_adam()
    |> Polaris.Updates.scale(learning_rate)
    |> then(fn optimizer ->
      {updates, _state} = optimizer.(gradients, %{}, params)
      Polaris.Updates.apply_updates(params, updates, %{})
    end)
  end

  @doc """
  Soft update of target network parameters.
  θ_target = τ * θ_local + (1 - τ) * θ_target
  """
  def soft_update(target_params, local_params, tau \\ 0.005) do
    Nx.Defn.jit(fn target, local, t ->
      target
      |> Nx.multiply(1.0 - t)
      |> Nx.add(Nx.multiply(local, t))
    end).(target_params, local_params, tau)
  end

  # Private helper functions

  defp build_hidden_layers(input, [], _activation), do: input
  defp build_hidden_layers(input, [size | rest], activation) do
    input
    |> Axon.dense(size, activation: activation)
    |> build_hidden_layers(rest, activation)
  end

  defnp gather_q_values(q_values, actions) do
    batch_size = Nx.axis_size(q_values, 0)
    batch_indices = Nx.iota({batch_size, 1})
    indices = Nx.concatenate([batch_indices, Nx.reshape(actions, {batch_size, 1})], axis: 1)
    Nx.gather(q_values, indices)
  end
end