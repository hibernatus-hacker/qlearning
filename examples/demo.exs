# QLearning Library Demo
# Run with: elixir -r examples/demo.exs

IO.puts """
=== QLearning Library Demo ===

This demo showcases the capabilities of the QLearning library,
including both tabular Q-learning and Deep Q-learning algorithms.
"""

# Example 1: Tabular Q-learning on GridWorld
IO.puts "\n1. Tabular Q-learning on GridWorld"
IO.puts "===================================="

# Define a simple GridWorld environment
defmodule SimpleGridWorld do
  @behaviour QLearning.Environment

  def get_states(), do: [{0,0}, {0,1}, {1,0}, {1,1}]
  def get_actions(), do: [:up, :down, :left, :right]
  def reset(), do: {0, 0}
  
  def step({x, y}, action) do
    new_pos = case action do
      :up -> {x, max(y - 1, 0)}
      :down -> {x, min(y + 1, 1)}
      :left -> {max(x - 1, 0), y}
      :right -> {min(x + 1, 1), y}
    end
    
    reward = if new_pos == {1, 1}, do: 10.0, else: -1.0
    done = new_pos == {1, 1}
    
    {new_pos, reward, done}
  end
end

# Train tabular Q-learning
hyperparams = %{alpha: 0.1, gamma: 0.9, epsilon: 0.1}
q_table = QLearning.train(SimpleGridWorld, hyperparams, 1000)

IO.puts "Sample Q-values after training:"
q_table
|> Enum.take(4)
|> Enum.each(fn {{state, action}, value} ->
  IO.puts "  Q(#{inspect(state)}, #{inspect(action)}) = #{Float.round(value, 3)}"
end)

# Example 2: Deep Q-learning on CartPole
IO.puts "\n2. Deep Q-learning on CartPole"
IO.puts "==============================="

try do
  # Quick DQN training
  IO.puts "Training DQN agent on CartPole..."
  
  hyperparams = %{
    learning_rate: 0.001,
    epsilon_start: 1.0,
    epsilon_end: 0.01,
    epsilon_decay: 1000,
    batch_size: 32
  }
  
  stats = QLearning.train_dqn(
    {4},  # State shape: [cart_pos, cart_vel, pole_angle, pole_angular_vel]
    2,    # Number of actions: left, right
    QLearning.Environments.CartPole,
    500,  # Episodes
    hyperparams
  )
  
  IO.puts "DQN Training completed!"
  IO.puts "Final stats: #{inspect(stats)}"
  
rescue
  error ->
    IO.puts "DQN training failed (this is expected in demo): #{inspect(error)}"
    IO.puts "This might be due to missing dependencies or runtime issues."
end

# Example 3: Policy comparison
IO.puts "\n3. Policy Exploration Strategies"
IO.puts "================================="

# Demonstrate different exploration policies
q_values = %{:action1 => 0.5, :action2 => 0.8, :action3 => 0.3}

IO.puts "Q-values: #{inspect(q_values)}"
IO.puts ""

# Epsilon-greedy
action_counts = %{:action1 => 10, :action2 => 5, :action3 => 15}

for policy <- [:epsilon_greedy, :greedy, :ucb, :boltzmann] do
  try do
    options = case policy do
      :epsilon_greedy -> %{epsilon: 0.2}
      :ucb -> %{action_counts: action_counts, total_steps: 30, c: 2.0}
      :boltzmann -> %{temperature: 1.0}
      _ -> %{}
    end
    
    action = QLearning.Policy.select_action(policy, q_values, options)
    IO.puts "#{policy} policy selected: #{action}"
  rescue
    _ -> IO.puts "#{policy} policy: Error (might need runtime dependencies)"
  end
end

# Example 4: Environment showcase
IO.puts "\n4. Available Environments"
IO.puts "========================="

IO.puts "GridWorld 4x4:"
try do
  grid_world = QLearning.Environments.GridWorld.simple_4x4()
  QLearning.Environments.GridWorld.print_grid(grid_world)
rescue
  _ -> IO.puts "GridWorld display error"
end

IO.puts "CartPole environment:"
IO.puts "- State space: 4D continuous (cart position, velocity, pole angle, angular velocity)"
IO.puts "- Action space: 2 discrete actions (left, right)"
IO.puts "- Goal: Balance pole for as long as possible"

# Example 5: Training utilities showcase
IO.puts "\n5. Training Utilities"
IO.puts "===================="

IO.puts "Available training functions:"
IO.puts "- QLearning.train/4: Tabular Q-learning"
IO.puts "- QLearning.train_dqn/5: Standard DQN"
IO.puts "- QLearning.train_double_dqn/5: Double DQN (reduces overestimation)"
IO.puts "- QLearning.train_dueling_dqn/5: Dueling DQN (better state value estimation)"
IO.puts ""
IO.puts "Trainer module provides:"
IO.puts "- Experiment tracking"
IO.puts "- Metrics collection"
IO.puts "- Data export capabilities"

IO.puts "\n=== Demo Complete ==="
IO.puts """
The QLearning library provides a comprehensive suite of reinforcement
learning algorithms suitable for both educational use and practical
applications. Key features include:

• Tabular Q-learning for discrete environments
• Deep Q-learning with experience replay and target networks
• Advanced DQN variants (Double DQN, Dueling DQN, Rainbow DQN)
• Multiple exploration strategies
• Built-in environments for testing
• Comprehensive training utilities and metrics tracking
• Built on Elixir/Nx/Axon for performance and reliability

Ready for your reinforcement learning experiments!
"""