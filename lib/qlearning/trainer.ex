defmodule QLearning.Trainer do
  @moduledoc """
  Training utilities and metrics tracking for Q-learning algorithms.
  
  Provides high-level training loops, experiment management, and 
  comprehensive metrics collection for both tabular and deep Q-learning.
  """

  use GenServer
  require Logger

  alias QLearning.{DQNAgent, Policy, Environment}

  defstruct [
    :experiment_name,
    :start_time,
    :episode_rewards,
    :episode_lengths,
    :loss_history,
    :epsilon_history,
    :evaluation_scores,
    :hyperparams,
    :total_episodes,
    :current_episode
  ]

  @type trainer_state :: %__MODULE__{
    experiment_name: String.t(),
    start_time: DateTime.t(),
    episode_rewards: [float()],
    episode_lengths: [integer()],
    loss_history: [float()],
    epsilon_history: [float()],
    evaluation_scores: [float()],
    hyperparams: map(),
    total_episodes: integer(),
    current_episode: integer()
  }

  # Client API

  @doc """
  Starts a new training session with metrics tracking.
  """
  def start_link(experiment_name, hyperparams \\ %{}, opts \\ []) do
    GenServer.start_link(__MODULE__, {experiment_name, hyperparams}, opts)
  end

  @doc """
  Trains a tabular Q-learning agent.
  """
  def train_tabular(trainer_pid, environment, hyperparams, episodes, opts \\ []) do
    GenServer.call(trainer_pid, {:train_tabular, environment, hyperparams, episodes, opts}, :infinity)
  end

  @doc """
  Trains a DQN agent.
  """
  def train_dqn(trainer_pid, agent_pid, environment, episodes, opts \\ []) do
    GenServer.call(trainer_pid, {:train_dqn, agent_pid, environment, episodes, opts}, :infinity)
  end

  @doc """
  Records metrics for an episode.
  """
  def record_episode(trainer_pid, episode_reward, episode_length, epsilon \\ nil) do
    GenServer.cast(trainer_pid, {:record_episode, episode_reward, episode_length, epsilon})
  end

  @doc """
  Records training loss.
  """
  def record_loss(trainer_pid, loss) do
    GenServer.cast(trainer_pid, {:record_loss, loss})
  end

  @doc """
  Evaluates the current agent and records the score.
  """
  def evaluate_and_record(trainer_pid, agent_pid, environment, episodes \\ 10) do
    GenServer.cast(trainer_pid, {:evaluate_and_record, agent_pid, environment, episodes})
  end

  @doc """
  Gets current training statistics.
  """
  def get_stats(trainer_pid) do
    GenServer.call(trainer_pid, :get_stats)
  end

  @doc """
  Exports training data to a file.
  """
  def export_data(trainer_pid, filename) do
    GenServer.call(trainer_pid, {:export_data, filename})
  end

  # GenServer Callbacks

  @impl true
  def init({experiment_name, hyperparams}) do
    state = %__MODULE__{
      experiment_name: experiment_name,
      start_time: DateTime.utc_now(),
      episode_rewards: [],
      episode_lengths: [],
      loss_history: [],
      epsilon_history: [],
      evaluation_scores: [],
      hyperparams: hyperparams,
      total_episodes: 0,
      current_episode: 0
    }
    
    Logger.info("Started training experiment: #{experiment_name}")
    {:ok, state}
  end

  @impl true
  def handle_call({:train_tabular, environment, hyperparams, episodes, opts}, _from, state) do
    Logger.info("Starting tabular Q-learning training for #{episodes} episodes")
    
    max_steps = Keyword.get(opts, :max_steps, 1000)
    eval_freq = Keyword.get(opts, :eval_frequency, 100)
    
    # Initialize Q-table
    states = Environment.get_states(environment)
    actions = Environment.get_actions(environment)
    q_table = QLearning.init_q_table(states, actions)
    
    final_q_table = 
      1..episodes
      |> Enum.reduce(q_table, fn episode, acc_q_table ->
        {new_q_table, episode_reward, episode_length} = 
          train_tabular_episode(environment, acc_q_table, hyperparams, max_steps)
        
        # Record metrics
        epsilon = Map.get(hyperparams, :epsilon, 0.1)
        record_episode(self(), episode_reward, episode_length, epsilon)
        
        # Evaluate periodically
        if rem(episode, eval_freq) == 0 do
          eval_score = Policy.evaluate_policy(environment, :greedy, new_q_table, 10)
          GenServer.cast(self(), {:record_evaluation, eval_score})
          Logger.info("Episode #{episode}, Avg Reward: #{Float.round(episode_reward, 2)}, Eval Score: #{Float.round(eval_score, 2)}")
        end
        
        new_q_table
      end)
    
    new_state = %{state | total_episodes: episodes, current_episode: episodes}
    {:reply, final_q_table, new_state}
  end

  @impl true
  def handle_call({:train_dqn, agent_pid, environment, episodes, opts}, _from, state) do
    Logger.info("Starting DQN training for #{episodes} episodes")
    
    max_steps = Keyword.get(opts, :max_steps, 500)
    eval_freq = Keyword.get(opts, :eval_frequency, 100)
    
    1..episodes
    |> Enum.each(fn episode ->
      {episode_reward, episode_length} = 
        train_dqn_episode(agent_pid, environment, max_steps)
      
      # Get agent stats
      agent_stats = DQNAgent.get_stats(agent_pid)
      
      # Record metrics
      record_episode(self(), episode_reward, episode_length, agent_stats.epsilon)
      
      # Evaluate periodically
      if rem(episode, eval_freq) == 0 do
        eval_score = evaluate_dqn_agent(agent_pid, environment, 10, max_steps)
        GenServer.cast(self(), {:record_evaluation, eval_score})
        Logger.info("Episode #{episode}, Reward: #{Float.round(episode_reward, 2)}, Steps: #{episode_length}, Epsilon: #{Float.round(agent_stats.epsilon, 3)}, Eval: #{Float.round(eval_score, 2)}")
      end
    end)
    
    new_state = %{state | total_episodes: episodes, current_episode: episodes}
    {:reply, :ok, new_state}
  end

  @impl true
  def handle_call(:get_stats, _from, state) do
    stats = calculate_training_stats(state)
    {:reply, stats, state}
  end

  @impl true
  def handle_call({:export_data, filename}, _from, state) do
    result = export_training_data(state, filename)
    {:reply, result, state}
  end

  @impl true
  def handle_cast({:record_episode, reward, length, epsilon}, state) do
    new_state = %{state |
      episode_rewards: [reward | state.episode_rewards],
      episode_lengths: [length | state.episode_lengths],
      epsilon_history: [epsilon | state.epsilon_history],
      current_episode: state.current_episode + 1
    }
    {:noreply, new_state}
  end

  @impl true
  def handle_cast({:record_loss, loss}, state) do
    new_state = %{state | loss_history: [loss | state.loss_history]}
    {:noreply, new_state}
  end

  @impl true
  def handle_cast({:record_evaluation, score}, state) do
    new_state = %{state | evaluation_scores: [score | state.evaluation_scores]}
    {:noreply, new_state}
  end

  @impl true
  def handle_cast({:evaluate_and_record, agent_pid, environment, episodes}, state) do
    score = evaluate_dqn_agent(agent_pid, environment, episodes, 500)
    new_state = %{state | evaluation_scores: [score | state.evaluation_scores]}
    {:noreply, new_state}
  end

  # Training implementations

  defp train_tabular_episode(environment, q_table, hyperparams, max_steps) do
    state = Environment.reset(environment)
    actions = Environment.get_actions(environment)
    
    alpha = Map.get(hyperparams, :alpha, 0.1)
    gamma = Map.get(hyperparams, :gamma, 0.9)
    epsilon = Map.get(hyperparams, :epsilon, 0.1)
    
    {final_q_table, episode_reward, step_count} = 
      1..max_steps
      |> Enum.reduce_while({q_table, 0.0, 0}, fn step, {acc_q_table, acc_reward, _count} ->
        action = QLearning.choose_action(state, acc_q_table, actions, epsilon)
        {next_state, reward, done} = Environment.step(environment, state, action)
        
        new_q_table = QLearning.update_q_value(
          acc_q_table, state, action, reward, next_state, actions, alpha, gamma
        )
        
        if done do
          {:halt, {new_q_table, acc_reward + reward, step}}
        else
          {:cont, {new_q_table, acc_reward + reward, step}}
        end
      end)
    
    {final_q_table, episode_reward, step_count}
  end

  defp train_dqn_episode(agent_pid, environment_module, max_steps) do
    # Start environment instance
    {:ok, env_pid} = apply(environment_module, :start_link, [[]])
    
    state = apply(environment_module, :reset, [env_pid])
    DQNAgent.reset_episode(agent_pid)
    
    {episode_reward, step_count} = 
      1..max_steps
      |> Enum.reduce_while({0.0, 0}, fn step, {acc_reward, _count} ->
        # Normalize state if it's CartPole
        normalized_state = 
          if environment_module == QLearning.Environments.CartPole do
            QLearning.Environments.CartPole.normalize_state(state)
          else
            state
          end
        
        action = DQNAgent.act(agent_pid, normalized_state)
        {next_state, reward, done} = apply(environment_module, :step, [env_pid, action])
        
        normalized_next_state = 
          if environment_module == QLearning.Environments.CartPole do
            QLearning.Environments.CartPole.normalize_state(next_state)
          else
            next_state
          end
        
        DQNAgent.remember(agent_pid, normalized_state, action, reward, normalized_next_state, done)
        DQNAgent.train_step(agent_pid)
        
        if done, do: GenServer.stop(env_pid)
        
        if done do
          {:halt, {acc_reward + reward, step}}
        else
          {:cont, {acc_reward + reward, step}}
        end
      end)
    
    # Clean up environment if still running
    if Process.alive?(env_pid), do: GenServer.stop(env_pid)
    
    {episode_reward, step_count}
  end

  defp evaluate_dqn_agent(agent_pid, environment_module, episodes, max_steps) do
    total_rewards = 
      1..episodes
      |> Enum.map(fn _episode ->
        {:ok, env_pid} = apply(environment_module, :start_link, [[]])
        
        state = apply(environment_module, :reset, [env_pid])
        
        episode_reward = 
          1..max_steps
          |> Enum.reduce_while(0.0, fn _step, acc_reward ->
            normalized_state = 
              if environment_module == QLearning.Environments.CartPole do
                QLearning.Environments.CartPole.normalize_state(state)
              else
                state
              end
            
            action = DQNAgent.act(agent_pid, normalized_state)
            {next_state, reward, done} = apply(environment_module, :step, [env_pid, action])
            
            if done, do: GenServer.stop(env_pid)
            
            if done do
              {:halt, acc_reward + reward}
            else
              {:cont, acc_reward + reward}
            end
          end)
        
        if Process.alive?(env_pid), do: GenServer.stop(env_pid)
        episode_reward
      end)
    
    Enum.sum(total_rewards) / episodes
  end

  defp calculate_training_stats(state) do
    episode_rewards = Enum.reverse(state.episode_rewards)
    episode_lengths = Enum.reverse(state.episode_lengths)
    epsilon_history = Enum.reverse(state.epsilon_history)
    loss_history = Enum.reverse(state.loss_history)
    
    %{
      experiment_name: state.experiment_name,
      hyperparams: state.hyperparams,
      start_time: state.start_time,
      total_episodes: state.current_episode,
      current_episode: state.current_episode,
      episode_rewards: episode_rewards,
      episode_lengths: episode_lengths,
      epsilon_history: epsilon_history,
      loss_history: loss_history,
      evaluation_scores: Enum.reverse(state.evaluation_scores),
      avg_reward: safe_avg(episode_rewards),
      avg_episode_length: safe_avg(episode_lengths),
      recent_avg_reward: safe_avg(Enum.take(episode_rewards, -100)),
      best_reward: safe_max(episode_rewards),
      worst_reward: safe_min(episode_rewards),
      current_epsilon: List.first(state.epsilon_history),
      total_training_time: DateTime.diff(DateTime.utc_now(), state.start_time)
    }
  end

  defp export_training_data(state, filename) do
    data = %{
      experiment_name: state.experiment_name,
      hyperparams: state.hyperparams,
      episode_rewards: Enum.reverse(state.episode_rewards),
      episode_lengths: Enum.reverse(state.episode_lengths),
      epsilon_history: Enum.reverse(state.epsilon_history),
      loss_history: Enum.reverse(state.loss_history),
      evaluation_scores: Enum.reverse(state.evaluation_scores),
      training_duration: DateTime.diff(DateTime.utc_now(), state.start_time)
    }
    
    case Jason.encode(data, pretty: true) do
      {:ok, json} ->
        File.write(filename, json)
        
      {:error, reason} ->
        {:error, "Failed to encode data: #{inspect(reason)}"}
    end
  end

  # Utility functions

  defp safe_avg([]), do: 0.0
  defp safe_avg(list), do: Enum.sum(list) / length(list)

  defp safe_max([]), do: 0.0
  defp safe_max(list), do: Enum.max(list)

  defp safe_min([]), do: 0.0
  defp safe_min(list), do: Enum.min(list)

  # Convenience functions for quick experiments

  @doc """
  Runs a complete tabular Q-learning experiment.
  """
  def run_tabular_experiment(experiment_name, environment, hyperparams, episodes, opts \\ []) do
    {:ok, trainer_pid} = start_link(experiment_name, hyperparams)
    
    q_table = train_tabular(trainer_pid, environment, hyperparams, episodes, opts)
    stats = get_stats(trainer_pid)
    
    GenServer.stop(trainer_pid)
    
    {q_table, stats}
  end

  @doc """
  Runs a complete DQN experiment.
  """
  def run_dqn_experiment(experiment_name, state_shape, num_actions, environment, hyperparams, episodes, opts \\ []) do
    {:ok, trainer_pid} = start_link(experiment_name, hyperparams)
    {:ok, agent_pid} = DQNAgent.start_link(state_shape, num_actions, hyperparams)
    
    train_dqn(trainer_pid, agent_pid, environment, episodes, opts)
    stats = get_stats(trainer_pid)
    
    GenServer.stop(trainer_pid)
    GenServer.stop(agent_pid)
    
    stats
  end
end