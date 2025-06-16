defmodule QLearning.DQNAgent do
  @moduledoc """
  Deep Q-Network (DQN) agent implementation.
  
  This agent uses a neural network to approximate Q-values and includes:
  - Experience replay for stable learning
  - Target network for stable Q-learning updates
  - Epsilon-greedy exploration strategy
  """

  use GenServer
  require Logger
  
  alias QLearning.{Network, ReplayBuffer}

  @type agent_state :: %{
    model: Axon.t(),
    params: map(),
    target_params: map(),
    buffer_pid: pid(),
    state_shape: tuple(),
    num_actions: pos_integer(),
    hyperparams: map(),
    step_count: non_neg_integer(),
    episode_count: non_neg_integer()
  }

  @type hyperparams :: %{
    learning_rate: float(),
    gamma: float(),
    epsilon_start: float(),
    epsilon_end: float(),
    epsilon_decay: pos_integer(),
    target_update_freq: pos_integer(),
    batch_size: pos_integer(),
    memory_size: pos_integer()
  }

  @default_hyperparams %{
    learning_rate: 0.001,
    gamma: 0.99,
    epsilon_start: 1.0,
    epsilon_end: 0.01,
    epsilon_decay: 1000,
    target_update_freq: 100,
    batch_size: 32,
    memory_size: 10_000
  }

  # Client API

  @doc """
  Starts a new DQN agent.
  
  ## Parameters
  - `state_shape`: Shape of state observations (e.g., {4})
  - `num_actions`: Number of possible actions
  - `hyperparams`: Training hyperparameters (optional)
  - `network_type`: :standard or :dueling (default: :standard)
  """
  def start_link(state_shape, num_actions, hyperparams \\ %{}, network_type \\ :standard, opts \\ []) do
    GenServer.start_link(__MODULE__, {state_shape, num_actions, hyperparams, network_type}, opts)
  end

  @doc """
  Selects an action given the current state.
  """
  def act(agent_pid, state) do
    GenServer.call(agent_pid, {:act, state})
  end

  @doc """
  Stores a transition in the replay buffer.
  """
  def remember(agent_pid, state, action, reward, next_state, done) do
    GenServer.cast(agent_pid, {:remember, state, action, reward, next_state, done})
  end

  @doc """
  Performs one training step if enough experiences are available.
  """
  def train_step(agent_pid) do
    GenServer.cast(agent_pid, :train_step)
  end

  @doc """
  Resets the agent for a new episode.
  """
  def reset_episode(agent_pid) do
    GenServer.cast(agent_pid, :reset_episode)
  end

  @doc """
  Gets current training statistics.
  """
  def get_stats(agent_pid) do
    GenServer.call(agent_pid, :get_stats)
  end

  @doc """
  Updates the target network parameters.
  """
  def update_target_network(agent_pid) do
    GenServer.cast(agent_pid, :update_target_network)
  end

  # Server Callbacks

  @impl true
  def init({state_shape, num_actions, hyperparams, network_type}) do
    merged_hyperparams = Map.merge(@default_hyperparams, hyperparams)
    
    # Create the neural network model
    state_size = Tuple.product(state_shape)
    model = case network_type do
      :dueling -> Network.build_dueling_dqn(state_size, num_actions)
      _ -> Network.build_dqn(state_size, num_actions)
    end
    
    # Initialize network parameters
    params = Network.init_params(model, state_shape)
    target_params = params
    
    # Start replay buffer
    {:ok, buffer_pid} = ReplayBuffer.start_link(
      merged_hyperparams.memory_size, 
      state_shape
    )
    
    state = %{
      model: model,
      params: params,
      target_params: target_params,
      buffer_pid: buffer_pid,
      state_shape: state_shape,
      num_actions: num_actions,
      hyperparams: merged_hyperparams,
      step_count: 0,
      episode_count: 0
    }
    
    Logger.info("DQN Agent initialized with #{network_type} network")
    {:ok, state}
  end

  @impl true
  def handle_call({:act, state}, _from, agent_state) do
    action = select_action(agent_state, state)
    {:reply, action, agent_state}
  end

  @impl true
  def handle_call(:get_stats, _from, agent_state) do
    buffer_size = ReplayBuffer.size(agent_state.buffer_pid)
    epsilon = calculate_epsilon(agent_state)
    
    stats = %{
      step_count: agent_state.step_count,
      episode_count: agent_state.episode_count,
      buffer_size: buffer_size,
      epsilon: epsilon
    }
    
    {:reply, stats, agent_state}
  end

  @impl true
  def handle_cast({:remember, state, action, reward, next_state, done}, agent_state) do
    transition = ReplayBuffer.create_transition(state, action, reward, next_state, done)
    ReplayBuffer.add(agent_state.buffer_pid, transition)
    {:noreply, agent_state}
  end

  @impl true
  def handle_cast(:train_step, agent_state) do
    new_state = perform_training_step(agent_state)
    {:noreply, new_state}
  end

  @impl true
  def handle_cast(:reset_episode, agent_state) do
    new_state = %{agent_state | episode_count: agent_state.episode_count + 1}
    {:noreply, new_state}
  end

  @impl true
  def handle_cast(:update_target_network, agent_state) do
    new_target_params = Network.soft_update(
      agent_state.target_params, 
      agent_state.params,
      0.005
    )
    
    new_state = %{agent_state | target_params: new_target_params}
    {:noreply, new_state}
  end

  # Private Functions

  defp select_action(agent_state, state) do
    epsilon = calculate_epsilon(agent_state)
    
    if :rand.uniform() < epsilon do
      # Exploration: random action
      :rand.uniform(agent_state.num_actions) - 1
    else
      # Exploitation: best action according to Q-network
      state_tensor = prepare_state_tensor(state, agent_state.state_shape)
      q_values = Network.predict(agent_state.model, agent_state.params, state_tensor)
      
      q_values
      |> Nx.argmax(axis: -1)
      |> Nx.to_number()
    end
  end

  defp calculate_epsilon(agent_state) do
    %{epsilon_start: start_eps, epsilon_end: end_eps, epsilon_decay: decay} = agent_state.hyperparams
    
    decay_rate = (start_eps - end_eps) / decay
    epsilon = start_eps - decay_rate * agent_state.step_count
    max(epsilon, end_eps)
  end

  defp perform_training_step(agent_state) do
    batch_size = agent_state.hyperparams.batch_size
    
    case ReplayBuffer.sample(agent_state.buffer_pid, batch_size) do
      nil ->
        # Not enough samples to train
        agent_state
        
      batch ->
        # Compute gradients and update parameters
        {_init_fn, predict_fn} = Axon.build(agent_state.model)
        
        loss_fn = fn params ->
          Network.q_learning_loss(
            params,
            predict_fn,
            batch.states,
            batch.actions,
            batch.rewards,
            batch.next_states,
            batch.dones,
            agent_state.hyperparams.gamma
          )
        end
        
        {_loss, gradients} = Nx.Defn.value_and_grad(loss_fn).(agent_state.params)
        
        new_params = Network.update_params(
          agent_state.params, 
          gradients, 
          agent_state.hyperparams.learning_rate
        )
        
        new_step_count = agent_state.step_count + 1
        
        # Update target network periodically
        new_target_params = 
          if rem(new_step_count, agent_state.hyperparams.target_update_freq) == 0 do
            Logger.debug("Updating target network at step #{new_step_count}")
            new_params
          else
            agent_state.target_params
          end
        
        %{agent_state |
          params: new_params,
          target_params: new_target_params,
          step_count: new_step_count
        }
    end
  end

  defp prepare_state_tensor(state, state_shape) when is_list(state) do
    state
    |> Nx.tensor()
    |> Nx.reshape({1, Tuple.product(state_shape)})
  end

  defp prepare_state_tensor(%Nx.Tensor{} = state, state_shape) do
    case Nx.shape(state) do
      ^state_shape -> Nx.reshape(state, {1, Tuple.product(state_shape)})
      _ -> state
    end
  end

  defp prepare_state_tensor(state, _state_shape) do
    Nx.tensor([state]) |> Nx.reshape({1, 1})
  end

  # Training utilities

  @doc """
  Trains the DQN agent on an environment for multiple episodes.
  """
  def train(agent_pid, environment, num_episodes, max_steps_per_episode \\ 1000) do
    Enum.each(1..num_episodes, fn episode ->
      run_episode(agent_pid, environment, max_steps_per_episode)
      
      if rem(episode, 100) == 0 do
        stats = get_stats(agent_pid)
        Logger.info("Episode #{episode}, Steps: #{stats.step_count}, Epsilon: #{Float.round(stats.epsilon, 3)}")
      end
    end)
  end

  defp run_episode(agent_pid, environment, max_steps) do
    state = QLearning.Environment.reset(environment)
    reset_episode(agent_pid)
    
    Enum.reduce_while(1..max_steps, state, fn _step, current_state ->
      action = act(agent_pid, current_state)
      {next_state, reward, done} = QLearning.Environment.step(environment, current_state, action)
      
      remember(agent_pid, current_state, action, reward, next_state, done)
      train_step(agent_pid)
      
      if done do
        {:halt, next_state}
      else
        {:cont, next_state}
      end
    end)
  end
end