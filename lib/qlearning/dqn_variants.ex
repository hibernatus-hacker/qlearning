defmodule QLearning.DQNVariants do
  @moduledoc """
  Advanced DQN variants including Double DQN, Dueling DQN, and Priority Experience Replay.
  
  These variants improve upon the basic DQN algorithm by addressing specific issues:
  - Double DQN: Reduces overestimation bias
  - Dueling DQN: Better state value estimation
  - Priority Experience Replay: Focuses learning on important experiences
  """

  use GenServer
  require Logger
  
  alias QLearning.{Network, ReplayBuffer}
  import Nx.Defn

  @type agent_type :: :double_dqn | :dueling_dqn | :rainbow_dqn
  @type agent_state :: %{
    agent_type: agent_type(),
    model: Axon.t(),
    params: map(),
    target_params: map(),
    buffer_pid: pid(),
    state_shape: tuple(),
    num_actions: pos_integer(),
    hyperparams: map(),
    step_count: non_neg_integer(),
    episode_count: non_neg_integer(),
    noisy_nets: boolean(),
    n_step: pos_integer()
  }

  @default_hyperparams %{
    learning_rate: 0.0001,
    gamma: 0.99,
    epsilon_start: 1.0,
    epsilon_end: 0.01,
    epsilon_decay: 10000,
    target_update_freq: 1000,
    batch_size: 32,
    memory_size: 100_000,
    tau: 0.005,  # For soft updates
    n_step: 3,   # For n-step returns
    priority_alpha: 0.6,  # Priority exponent
    priority_beta_start: 0.4,
    priority_beta_frames: 100_000
  }

  # Client API

  @doc """
  Starts a new DQN variant agent.
  
  ## Parameters
  - `agent_type`: :double_dqn, :dueling_dqn, or :rainbow_dqn
  - `state_shape`: Shape of state observations
  - `num_actions`: Number of possible actions
  - `hyperparams`: Training hyperparameters
  """
  def start_link(agent_type, state_shape, num_actions, hyperparams \\ %{}, opts \\ []) do
    GenServer.start_link(__MODULE__, {agent_type, state_shape, num_actions, hyperparams}, opts)
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
  Performs one training step.
  """
  def train_step(agent_pid) do
    GenServer.cast(agent_pid, :train_step)
  end

  @doc """
  Gets current training statistics.
  """
  def get_stats(agent_pid) do
    GenServer.call(agent_pid, :get_stats)
  end

  # GenServer Callbacks

  @impl true
  def init({agent_type, state_shape, num_actions, hyperparams}) do
    merged_hyperparams = Map.merge(@default_hyperparams, hyperparams)
    
    # Create the appropriate network architecture
    state_size = Tuple.product(state_shape)
    model = case agent_type do
      :dueling_dqn -> Network.build_dueling_dqn(state_size, num_actions)
      :rainbow_dqn -> build_rainbow_network(state_size, num_actions)
      _ -> Network.build_dqn(state_size, num_actions)
    end
    
    # Initialize parameters
    params = Network.init_params(model, state_shape)
    target_params = params
    
    # Start appropriate replay buffer
    buffer_module = case agent_type do
      :rainbow_dqn -> PriorityReplayBuffer
      _ -> ReplayBuffer
    end
    
    {:ok, buffer_pid} = buffer_module.start_link(
      merged_hyperparams.memory_size,
      state_shape
    )
    
    state = %{
      agent_type: agent_type,
      model: model,
      params: params,
      target_params: target_params,
      buffer_pid: buffer_pid,
      state_shape: state_shape,
      num_actions: num_actions,
      hyperparams: merged_hyperparams,
      step_count: 0,
      episode_count: 0,
      noisy_nets: agent_type == :rainbow_dqn,
      n_step: merged_hyperparams.n_step
    }
    
    Logger.info("#{agent_type} Agent initialized")
    {:ok, state}
  end

  @impl true
  def handle_call({:act, state}, _from, agent_state) do
    action = select_action(agent_state, state)
    {:reply, action, agent_state}
  end

  @impl true
  def handle_call(:get_stats, _from, agent_state) do
    buffer_size = get_buffer_size(agent_state.buffer_pid, agent_state.agent_type)
    epsilon = calculate_epsilon(agent_state)
    
    stats = %{
      agent_type: agent_state.agent_type,
      step_count: agent_state.step_count,
      episode_count: agent_state.episode_count,
      buffer_size: buffer_size,
      epsilon: epsilon
    }
    
    {:reply, stats, agent_state}
  end

  @impl true
  def handle_cast({:remember, state, action, reward, next_state, done}, agent_state) do
    transition = create_transition(state, action, reward, next_state, done)
    add_to_buffer(agent_state.buffer_pid, agent_state.agent_type, transition)
    {:noreply, agent_state}
  end

  @impl true
  def handle_cast(:train_step, agent_state) do
    new_state = perform_training_step(agent_state)
    {:noreply, new_state}
  end

  # Action selection

  defp select_action(agent_state, state) do
    if agent_state.noisy_nets do
      # Noisy networks for exploration
      select_noisy_action(agent_state, state)
    else
      # Epsilon-greedy exploration
      epsilon = calculate_epsilon(agent_state)
      select_epsilon_greedy_action(agent_state, state, epsilon)
    end
  end

  defp select_epsilon_greedy_action(agent_state, state, epsilon) do
    if :rand.uniform() < epsilon do
      :rand.uniform(agent_state.num_actions) - 1
    else
      state_tensor = prepare_state_tensor(state, agent_state.state_shape)
      q_values = Network.predict(agent_state.model, agent_state.params, state_tensor)
      
      q_values
      |> Nx.argmax(axis: -1)
      |> Nx.to_number()
    end
  end

  defp select_noisy_action(agent_state, state) do
    state_tensor = prepare_state_tensor(state, agent_state.state_shape)
    q_values = Network.predict(agent_state.model, agent_state.params, state_tensor)
    
    q_values
    |> Nx.argmax(axis: -1)
    |> Nx.to_number()
  end

  # Training step implementations

  defp perform_training_step(agent_state) do
    batch_size = agent_state.hyperparams.batch_size
    
    case sample_batch(agent_state.buffer_pid, agent_state.agent_type, batch_size) do
      nil ->
        agent_state
        
      batch ->
        case agent_state.agent_type do
          :double_dqn -> train_double_dqn(agent_state, batch)
          :dueling_dqn -> train_dueling_dqn(agent_state, batch)
          :rainbow_dqn -> train_rainbow_dqn(agent_state, batch)
          _ -> train_standard_dqn(agent_state, batch)
        end
    end
  end

  defp train_double_dqn(agent_state, batch) do
    {_init_fn, predict_fn} = Axon.build(agent_state.model)
    
    loss_fn = fn params ->
      double_dqn_loss(
        params,
        agent_state.target_params,
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
    
    update_agent_after_training(agent_state, new_params)
  end

  defp train_dueling_dqn(agent_state, batch) do
    # Dueling DQN uses the same loss as standard DQN but with different network architecture
    train_standard_dqn(agent_state, batch)
  end

  defp train_rainbow_dqn(agent_state, batch) do
    # Rainbow DQN combines multiple improvements
    {_init_fn, predict_fn} = Axon.build(agent_state.model)
    
    loss_fn = fn params ->
      rainbow_loss(
        params,
        agent_state.target_params,
        predict_fn,
        batch.states,
        batch.actions,
        batch.rewards,
        batch.next_states,
        batch.dones,
        batch.priorities,
        agent_state.hyperparams.gamma,
        agent_state.n_step
      )
    end
    
    {_loss, gradients} = Nx.Defn.value_and_grad(loss_fn).(agent_state.params)
    
    new_params = Network.update_params(
      agent_state.params,
      gradients,
      agent_state.hyperparams.learning_rate
    )
    
    update_agent_after_training(agent_state, new_params)
  end

  defp train_standard_dqn(agent_state, batch) do
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
    
    update_agent_after_training(agent_state, new_params)
  end

  # Loss functions for variants

  defn double_dqn_loss(params, target_params, model_fn, states, actions, rewards, next_states, dones, gamma) do
    # Current Q-values
    current_q_values = model_fn.(params, %{"state" => states})
    current_q = gather_q_values(current_q_values, actions)
    
    # Double DQN: Use online network to select actions, target network to evaluate
    next_q_values_online = model_fn.(params, %{"state" => next_states})
    next_actions = Nx.argmax(next_q_values_online, axis: -1)
    
    next_q_values_target = Nx.Defn.stop_grad(model_fn.(target_params, %{"state" => next_states}))
    next_q = gather_q_values(next_q_values_target, next_actions)
    
    # Target: r + γ * Q_target(s', argmax_a Q_online(s', a)) * (1 - done)
    targets = Nx.add(rewards, Nx.multiply(gamma, Nx.multiply(next_q, Nx.subtract(1.0, dones))))
    
    Network.huber_loss(targets, current_q)
  end

  defn rainbow_loss(params, target_params, model_fn, states, actions, rewards, next_states, dones, priorities, gamma, n_step) do
    # Simplified Rainbow loss (would include distributional RL in full implementation)
    double_dqn_loss(params, target_params, model_fn, states, actions, rewards, next_states, dones, gamma)
  end

  defnp gather_q_values(q_values, actions) do
    batch_size = Nx.axis_size(q_values, 0)
    batch_indices = Nx.iota({batch_size, 1})
    indices = Nx.concatenate([batch_indices, Nx.reshape(actions, {batch_size, 1})], axis: 1)
    Nx.gather(q_values, indices)
  end

  # Helper functions

  defp update_agent_after_training(agent_state, new_params) do
    new_step_count = agent_state.step_count + 1
    
    # Update target network
    new_target_params = 
      if rem(new_step_count, agent_state.hyperparams.target_update_freq) == 0 do
        Logger.debug("Updating target network at step #{new_step_count}")
        Network.soft_update(
          agent_state.target_params,
          new_params,
          agent_state.hyperparams.tau
        )
      else
        agent_state.target_params
      end
    
    %{agent_state |
      params: new_params,
      target_params: new_target_params,
      step_count: new_step_count
    }
  end

  defp calculate_epsilon(agent_state) do
    if agent_state.noisy_nets do
      0.0  # No epsilon when using noisy networks
    else
      %{epsilon_start: start_eps, epsilon_end: end_eps, epsilon_decay: decay} = agent_state.hyperparams
      
      decay_rate = (start_eps - end_eps) / decay
      epsilon = start_eps - decay_rate * agent_state.step_count
      max(epsilon, end_eps)
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

  # Buffer interface functions

  defp get_buffer_size(buffer_pid, :rainbow_dqn) do
    PriorityReplayBuffer.size(buffer_pid)
  end

  defp get_buffer_size(buffer_pid, _agent_type) do
    ReplayBuffer.size(buffer_pid)
  end

  defp add_to_buffer(buffer_pid, :rainbow_dqn, transition) do
    PriorityReplayBuffer.add(buffer_pid, transition)
  end

  defp add_to_buffer(buffer_pid, _agent_type, transition) do
    ReplayBuffer.add(buffer_pid, transition)
  end

  defp sample_batch(buffer_pid, :rainbow_dqn, batch_size) do
    PriorityReplayBuffer.sample(buffer_pid, batch_size)
  end

  defp sample_batch(buffer_pid, _agent_type, batch_size) do
    case ReplayBuffer.sample(buffer_pid, batch_size) do
      nil -> nil
      batch -> Map.put(batch, :priorities, Nx.broadcast(1.0, {batch_size}))
    end
  end

  defp create_transition(state, action, reward, next_state, done) do
    ReplayBuffer.create_transition(state, action, reward, next_state, done)
  end

  # Network architectures

  defp build_rainbow_network(input_size, output_size) do
    # Simplified Rainbow network (full version would include distributional head)
    Network.build_dueling_dqn(input_size, output_size, [512, 512], :relu)
  end

  # Priority Replay Buffer (simplified implementation)
  defmodule PriorityReplayBuffer do
    use GenServer
    
    def start_link(capacity, state_shape, opts \\ []) do
      # For now, use regular replay buffer
      # Full implementation would include priority sampling
      ReplayBuffer.start_link(capacity, state_shape, opts)
    end
    
    def init(init_arg) do
      {:ok, init_arg}
    end
    
    def add(buffer_pid, transition) do
      ReplayBuffer.add(buffer_pid, transition)
    end
    
    def sample(buffer_pid, batch_size) do
      case ReplayBuffer.sample(buffer_pid, batch_size) do
        nil -> nil
        batch -> 
          # Add dummy priorities
          priorities = Nx.broadcast(1.0, {batch_size})
          Map.put(batch, :priorities, priorities)
      end
    end
    
    def size(buffer_pid) do
      ReplayBuffer.size(buffer_pid)
    end
  end

  # Training utilities

  @doc """
  Trains a DQN variant on an environment.
  """
  def train(agent_pid, environment, num_episodes, max_steps_per_episode \\ 500) do
    Enum.each(1..num_episodes, fn episode ->
      run_episode(agent_pid, environment, max_steps_per_episode)
      
      if rem(episode, 100) == 0 do
        stats = get_stats(agent_pid)
        Logger.info("Episode #{episode}, Agent: #{stats.agent_type}, Steps: #{stats.step_count}, Epsilon: #{Float.round(stats.epsilon, 3)}")
      end
    end)
  end

  defp run_episode(agent_pid, environment, max_steps) do
    {:ok, env_pid} = apply(environment, :start_link, [[]])
    state = apply(environment, :reset, [env_pid])
    
    Enum.reduce_while(1..max_steps, state, fn _step, current_state ->
      normalized_state = 
        if environment == QLearning.Environments.CartPole do
          QLearning.Environments.CartPole.normalize_state(current_state)
        else
          current_state
        end
      
      action = act(agent_pid, normalized_state)
      {next_state, reward, done} = apply(environment, :step, [env_pid, action])
      
      normalized_next_state = 
        if environment == QLearning.Environments.CartPole do
          QLearning.Environments.CartPole.normalize_state(next_state)
        else
          next_state
        end
      
      remember(agent_pid, normalized_state, action, reward, normalized_next_state, done)
      train_step(agent_pid)
      
      if done, do: GenServer.stop(env_pid)
      
      if done do
        {:halt, next_state}
      else
        {:cont, next_state}
      end
    end)
    
    # Clean up if environment still running
    if Process.alive?(env_pid), do: GenServer.stop(env_pid)
  end
end