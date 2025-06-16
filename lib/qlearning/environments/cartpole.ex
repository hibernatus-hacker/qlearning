defmodule QLearning.Environments.CartPole do
  @moduledoc """
  A simplified CartPole environment for testing Deep Q-learning.
  
  The agent balances a pole on a cart by moving left or right.
  - State: [cart_position, cart_velocity, pole_angle, pole_angular_velocity]
  - Actions: 0 (left), 1 (right)
  - Rewards: +1 for each step the pole remains upright
  - Episode ends when pole angle exceeds ±12° or cart moves ±2.4 units
  """

  use GenServer
  import Nx.Defn

  @behaviour QLearning.Environment

  defstruct [
    :cart_position,
    :cart_velocity, 
    :pole_angle,
    :pole_angular_velocity,
    :step_count,
    :max_steps
  ]

  @type cartpole_state :: %__MODULE__{
    cart_position: float(),
    cart_velocity: float(),
    pole_angle: float(),
    pole_angular_velocity: float(),
    step_count: integer(),
    max_steps: integer()
  }

  # Physics constants
  @gravity 9.8
  @cart_mass 1.0
  @pole_mass 0.1
  @total_mass @cart_mass + @pole_mass
  @pole_length 0.5
  @pole_mass_length @pole_mass * @pole_length
  @force_magnitude 10.0
  @tau 0.02  # Time step

  # Thresholds
  @theta_threshold_radians 12 * 2 * :math.pi() / 360
  @x_threshold 2.4

  @actions [0, 1]  # 0: left, 1: right

  # Environment API

  def start_link(opts \\ []) do
    max_steps = Keyword.get(opts, :max_steps, 500)
    GenServer.start_link(__MODULE__, %{max_steps: max_steps}, opts)
  end

  def reset(pid) do
    GenServer.call(pid, :reset)
  end

  def step(pid, action) when is_pid(pid) do
    GenServer.call(pid, {:step, action})
  end

  def get_state(pid) do
    GenServer.call(pid, :get_state)
  end

  # Behaviour implementation (for compatibility)

  @impl true
  def get_states() do
    # Continuous state space - return sample states for discrete Q-learning
    # In practice, you'd discretize the state space
    []
  end

  @impl true
  def get_actions(), do: @actions

  @impl true
  def reset() do
    # Return initial state as list
    [0.0, 0.0, 0.0, 0.0]
  end

  @impl true
  def step(state, action) when is_list(state) do
    # Convert list state to struct for processing
    cartpole_state = %__MODULE__{
      cart_position: Enum.at(state, 0),
      cart_velocity: Enum.at(state, 1),
      pole_angle: Enum.at(state, 2),
      pole_angular_velocity: Enum.at(state, 3),
      step_count: 0,
      max_steps: 500
    }
    
    new_state = update_physics(cartpole_state, action)
    {reward, done} = calculate_reward_and_done(new_state)
    
    next_state_list = [
      new_state.cart_position,
      new_state.cart_velocity,
      new_state.pole_angle,
      new_state.pole_angular_velocity
    ]
    
    {next_state_list, reward, done}
  end

  # GenServer callbacks

  @impl true
  def init(%{max_steps: max_steps}) do
    state = %__MODULE__{
      cart_position: random_initial_value(0.05),
      cart_velocity: random_initial_value(0.05),
      pole_angle: random_initial_value(0.05),
      pole_angular_velocity: random_initial_value(0.05),
      step_count: 0,
      max_steps: max_steps
    }
    
    {:ok, state}
  end

  @impl true
  def handle_call(:reset, _from, state) do
    new_state = %__MODULE__{
      cart_position: random_initial_value(0.05),
      cart_velocity: random_initial_value(0.05),
      pole_angle: random_initial_value(0.05),
      pole_angular_velocity: random_initial_value(0.05),
      step_count: 0,
      max_steps: state.max_steps
    }
    
    state_vector = state_to_vector(new_state)
    {:reply, state_vector, new_state}
  end

  @impl true
  def handle_call({:step, action}, _from, state) do
    new_state = update_physics(state, action)
    new_state = %{new_state | step_count: new_state.step_count + 1}
    
    {reward, done} = calculate_reward_and_done(new_state)
    next_state_vector = state_to_vector(new_state)
    
    {:reply, {next_state_vector, reward, done}, new_state}
  end

  @impl true
  def handle_call(:get_state, _from, state) do
    state_vector = state_to_vector(state)
    {:reply, state_vector, state}
  end

  # Physics simulation

  defnp update_physics_defn(
    cart_position, cart_velocity, pole_angle, pole_angular_velocity, force
  ) do
    costheta = Nx.cos(pole_angle)
    sintheta = Nx.sin(pole_angle)
    
    temp = (force + @pole_mass_length * pole_angular_velocity * pole_angular_velocity * sintheta) / @total_mass
    
    thetaacc = (@gravity * sintheta - costheta * temp) / 
               (@pole_length * (4.0/3.0 - @pole_mass * costheta * costheta / @total_mass))
    
    xacc = temp - @pole_mass_length * thetaacc * costheta / @total_mass
    
    # Update state using Euler integration
    new_cart_velocity = cart_velocity + @tau * xacc
    new_cart_position = cart_position + @tau * new_cart_velocity
    new_pole_angular_velocity = pole_angular_velocity + @tau * thetaacc
    new_pole_angle = pole_angle + @tau * new_pole_angular_velocity
    
    {new_cart_position, new_cart_velocity, new_pole_angle, new_pole_angular_velocity}
  end

  defp update_physics(state, action) do
    force = if action == 1, do: @force_magnitude, else: -@force_magnitude
    
    {new_cart_position, new_cart_velocity, new_pole_angle, new_pole_angular_velocity} = 
      update_physics_defn(
        state.cart_position,
        state.cart_velocity, 
        state.pole_angle,
        state.pole_angular_velocity,
        force
      )
      |> then(fn result ->
        result
        |> Tuple.to_list()
        |> Enum.map(&Nx.to_number/1)
        |> List.to_tuple()
      end)
    
    %{state |
      cart_position: new_cart_position,
      cart_velocity: new_cart_velocity,
      pole_angle: new_pole_angle,
      pole_angular_velocity: new_pole_angular_velocity
    }
  end

  defp calculate_reward_and_done(state) do
    done = state.cart_position < -@x_threshold or
           state.cart_position > @x_threshold or
           state.pole_angle < -@theta_threshold_radians or
           state.pole_angle > @theta_threshold_radians or
           state.step_count >= state.max_steps
    
    reward = if done, do: 0.0, else: 1.0
    
    {reward, done}
  end

  defp state_to_vector(state) do
    [
      state.cart_position,
      state.cart_velocity,
      state.pole_angle,
      state.pole_angular_velocity
    ]
  end

  defp random_initial_value(range) do
    (:rand.uniform() - 0.5) * 2 * range
  end

  # Utility functions

  @doc """
  Creates a simple wrapper for the stateless Environment behaviour.
  """
  def create_stateless_env() do
    QLearning.Environments.CartPole
  end

  @doc """
  Normalizes the state for neural network input.
  """
  def normalize_state(state) when is_list(state) do
    # Normalize each component to roughly [-1, 1] range
    [cart_pos, cart_vel, pole_angle, pole_angular_vel] = state
    
    [
      cart_pos / @x_threshold,
      cart_vel / 2.0,
      pole_angle / @theta_threshold_radians,
      pole_angular_vel / 2.0
    ]
  end

  @doc """
  Returns the observation space dimensions.
  """
  def observation_space_size(), do: 4

  @doc """
  Returns the action space size.
  """
  def action_space_size(), do: 2

  @doc """
  Trains a DQN agent on the CartPole environment.
  """
  def train_dqn_agent(agent_pid, num_episodes \\ 1000) do
    Enum.each(1..num_episodes, fn episode ->
      {:ok, env_pid} = start_link()
      
      state = reset(env_pid)
      total_reward = run_episode(agent_pid, env_pid, state, 0)
      
      GenServer.stop(env_pid)
      
      if rem(episode, 100) == 0 do
        stats = QLearning.DQNAgent.get_stats(agent_pid)
        IO.puts("Episode #{episode}, Total Reward: #{total_reward}, Epsilon: #{Float.round(stats.epsilon, 3)}")
      end
    end)
  end

  defp run_episode(agent_pid, env_pid, state, total_reward) do
    normalized_state = normalize_state(state)
    action = QLearning.DQNAgent.act(agent_pid, normalized_state)
    
    {next_state, reward, done} = step(env_pid, action)
    normalized_next_state = normalize_state(next_state)
    
    QLearning.DQNAgent.remember(
      agent_pid, 
      normalized_state, 
      action, 
      reward, 
      normalized_next_state, 
      done
    )
    
    QLearning.DQNAgent.train_step(agent_pid)
    
    if done do
      QLearning.DQNAgent.reset_episode(agent_pid)
      total_reward + reward
    else
      run_episode(agent_pid, env_pid, next_state, total_reward + reward)
    end
  end
end