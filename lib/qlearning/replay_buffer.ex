defmodule QLearning.ReplayBuffer do
  @moduledoc """
  Experience replay buffer for Deep Q-learning.
  
  Stores transitions (state, action, reward, next_state, done) and provides
  efficient random sampling for training. Uses Nx tensors for efficient
  storage and batch operations.
  """

  use GenServer
  require Logger

  @type transition :: %{
    state: Nx.Tensor.t(),
    action: integer(),
    reward: float(),
    next_state: Nx.Tensor.t(),
    done: boolean()
  }

  @type buffer_state :: %{
    states: Nx.Tensor.t(),
    actions: Nx.Tensor.t(),
    rewards: Nx.Tensor.t(),
    next_states: Nx.Tensor.t(),
    dones: Nx.Tensor.t(),
    capacity: pos_integer(),
    size: non_neg_integer(),
    position: non_neg_integer(),
    state_shape: tuple()
  }

  # Client API

  @doc """
  Starts a new replay buffer process.
  
  ## Parameters
  - `capacity`: Maximum number of transitions to store
  - `state_shape`: Shape of state tensors (e.g., {4} for 4D state)
  - `opts`: GenServer options
  """
  def start_link(capacity, state_shape, opts \\ []) do
    GenServer.start_link(__MODULE__, {capacity, state_shape}, opts)
  end

  @doc """
  Adds a transition to the buffer.
  """
  def add(buffer_pid, transition) do
    GenServer.cast(buffer_pid, {:add, transition})
  end

  @doc """
  Samples a batch of transitions from the buffer.
  Returns nil if buffer doesn't have enough samples.
  """
  def sample(buffer_pid, batch_size) do
    GenServer.call(buffer_pid, {:sample, batch_size})
  end

  @doc """
  Returns the current size of the buffer.
  """
  def size(buffer_pid) do
    GenServer.call(buffer_pid, :size)
  end

  @doc """
  Returns the capacity of the buffer.
  """
  def capacity(buffer_pid) do
    GenServer.call(buffer_pid, :capacity)
  end

  @doc """
  Checks if buffer has enough samples for training.
  """
  def is_ready(buffer_pid, min_size) do
    GenServer.call(buffer_pid, {:is_ready, min_size})
  end

  @doc """
  Clears all transitions from the buffer.
  """
  def clear(buffer_pid) do
    GenServer.cast(buffer_pid, :clear)
  end

  # Server Callbacks

  @impl true
  def init({capacity, state_shape}) do
    state_size = Tuple.product(state_shape)
    
    initial_state = %{
      states: Nx.broadcast(0.0, {capacity, state_size}),
      actions: Nx.broadcast(0, {capacity}),
      rewards: Nx.broadcast(0.0, {capacity}),
      next_states: Nx.broadcast(0.0, {capacity, state_size}),
      dones: Nx.broadcast(0.0, {capacity}),
      capacity: capacity,
      size: 0,
      position: 0,
      state_shape: state_shape
    }
    
    {:ok, initial_state}
  end

  @impl true
  def handle_cast({:add, transition}, state) do
    new_state = add_transition(state, transition)
    {:noreply, new_state}
  end

  @impl true
  def handle_cast(:clear, state) do
    state_size = Tuple.product(state.state_shape)
    
    cleared_state = %{state |
      states: Nx.broadcast(0.0, {state.capacity, state_size}),
      actions: Nx.broadcast(0, {state.capacity}),
      rewards: Nx.broadcast(0.0, {state.capacity}),
      next_states: Nx.broadcast(0.0, {state.capacity, state_size}),
      dones: Nx.broadcast(0.0, {state.capacity}),
      size: 0,
      position: 0
    }
    
    {:noreply, cleared_state}
  end

  @impl true
  def handle_call({:sample, batch_size}, _from, state) do
    if state.size < batch_size do
      {:reply, nil, state}
    else
      batch = sample_batch(state, batch_size)
      {:reply, batch, state}
    end
  end

  @impl true
  def handle_call(:size, _from, state) do
    {:reply, state.size, state}
  end

  def handle_call(:capacity, _from, state) do
    {:reply, state.capacity, state}
  end

  def handle_call({:is_ready, min_size}, _from, state) do
    {:reply, state.size >= min_size, state}
  end

  # Private Functions

  defp add_transition(state, transition) do
    pos = state.position
    
    # Flatten state tensors for storage
    flat_state = Nx.flatten(transition.state)
    flat_next_state = Nx.flatten(transition.next_state)
    
    # Update tensors at position
    new_states = Nx.put_slice(state.states, [pos, 0], Nx.reshape(flat_state, {1, :auto}))
    new_actions = Nx.put_slice(state.actions, [pos], Nx.tensor([transition.action]))
    new_rewards = Nx.put_slice(state.rewards, [pos], Nx.tensor([transition.reward]))
    new_next_states = Nx.put_slice(state.next_states, [pos, 0], Nx.reshape(flat_next_state, {1, :auto}))
    new_dones = Nx.put_slice(state.dones, [pos], Nx.tensor([(if transition.done, do: 1.0, else: 0.0)]))
    
    new_position = rem(pos + 1, state.capacity)
    new_size = min(state.size + 1, state.capacity)
    
    %{state |
      states: new_states,
      actions: new_actions,
      rewards: new_rewards,
      next_states: new_next_states,
      dones: new_dones,
      position: new_position,
      size: new_size
    }
  end

  defp sample_batch(state, batch_size) do
    # Generate random indices
    indices = 
      0..(state.size - 1)
      |> Enum.to_list()
      |> Enum.take_random(batch_size)
      |> Nx.tensor()
    
    # Sample from buffers
    sampled_states = Nx.take(state.states, indices, axis: 0)
    sampled_actions = Nx.take(state.actions, indices)
    sampled_rewards = Nx.take(state.rewards, indices)
    sampled_next_states = Nx.take(state.next_states, indices, axis: 0)
    sampled_dones = Nx.take(state.dones, indices)
    
    # Reshape states back to original shape
    batch_state_shape = Tuple.insert_at(state.state_shape, 0, batch_size)
    
    %{
      states: Nx.reshape(sampled_states, batch_state_shape),
      actions: sampled_actions,
      rewards: sampled_rewards,
      next_states: Nx.reshape(sampled_next_states, batch_state_shape),
      dones: sampled_dones
    }
  end

  # Utility functions for creating transitions

  @doc """
  Creates a transition map from individual components.
  """
  def create_transition(state, action, reward, next_state, done) do
    %{
      state: ensure_tensor(state),
      action: action,
      reward: Float.round(reward, 6),
      next_state: ensure_tensor(next_state),
      done: done
    }
  end

  defp ensure_tensor(data) when is_list(data), do: Nx.tensor(data)
  defp ensure_tensor(%Nx.Tensor{} = tensor), do: tensor
  defp ensure_tensor(data), do: Nx.tensor([data])
end