defmodule QLearning.PolicyTest do
  use ExUnit.Case
  alias QLearning.Policy

  describe "epsilon_greedy policy" do
    test "selects best action when epsilon is 0" do
      q_values = %{:action1 => 0.5, :action2 => 0.8, :action3 => 0.2}
      action = Policy.select_action(:epsilon_greedy, q_values, %{epsilon: 0.0})
      assert action == :action2
    end

    test "selects random action sometimes when epsilon > 0" do
      q_values = %{:action1 => 0.5, :action2 => 0.8, :action3 => 0.2}
      
      # Run multiple times to test randomness
      actions = Enum.map(1..100, fn _ ->
        Policy.select_action(:epsilon_greedy, q_values, %{epsilon: 1.0})
      end)
      
      # With epsilon=1.0, should see all actions
      unique_actions = Enum.uniq(actions)
      assert length(unique_actions) > 1
    end

    test "works with Nx tensor Q-values" do
      q_values = Nx.tensor([0.5, 0.8, 0.2])
      action = Policy.select_action(:epsilon_greedy, q_values, %{epsilon: 0.0})
      assert action == 1  # Index of highest value
    end
  end

  describe "greedy policy" do
    test "always selects best action with map Q-values" do
      q_values = %{:action1 => 0.5, :action2 => 0.8, :action3 => 0.2}
      action = Policy.select_action(:greedy, q_values)
      assert action == :action2
    end

    test "always selects best action with tensor Q-values" do
      q_values = Nx.tensor([0.2, 0.8, 0.5])
      action = Policy.select_action(:greedy, q_values)
      assert action == 1
    end

    test "handles equal Q-values consistently" do
      q_values = %{:action1 => 0.5, :action2 => 0.5}
      action = Policy.select_action(:greedy, q_values)
      assert action in [:action1, :action2]
    end
  end

  describe "boltzmann policy" do
    test "selects actions probabilistically based on Q-values" do
      q_values = %{:action1 => 1.0, :action2 => 2.0, :action3 => 0.5}
      
      # Run multiple times to check probabilistic behavior
      actions = Enum.map(1..100, fn _ ->
        Policy.select_action(:boltzmann, q_values, %{temperature: 1.0})
      end)
      
      unique_actions = Enum.uniq(actions)
      assert length(unique_actions) > 1
      
      # Higher Q-value should be selected more often
      action2_count = Enum.count(actions, &(&1 == :action2))
      action3_count = Enum.count(actions, &(&1 == :action3))
      assert action2_count > action3_count
    end

    test "works with tensor Q-values" do
      q_values = Nx.tensor([1.0, 2.0, 0.5])
      action = Policy.select_action(:boltzmann, q_values, %{temperature: 1.0})
      assert action in [0, 1, 2]
    end

    test "temperature affects exploration" do
      q_values = %{:action1 => 1.0, :action2 => 2.0}
      
      # Low temperature should favor exploitation
      actions_low_temp = Enum.map(1..100, fn _ ->
        Policy.select_action(:boltzmann, q_values, %{temperature: 0.1})
      end)
      
      action2_count_low = Enum.count(actions_low_temp, &(&1 == :action2))
      
      # High temperature should be more exploratory
      actions_high_temp = Enum.map(1..100, fn _ ->
        Policy.select_action(:boltzmann, q_values, %{temperature: 10.0})
      end)
      
      action2_count_high = Enum.count(actions_high_temp, &(&1 == :action2))
      
      # Lower temperature should select best action more often
      assert action2_count_low > action2_count_high
    end
  end

  describe "UCB policy" do
    test "selects unvisited actions first" do
      q_values = %{:action1 => 0.5, :action2 => 0.8, :action3 => 0.2}
      action_counts = %{:action1 => 5, :action2 => 10}  # :action3 unvisited
      
      action = Policy.select_action(:ucb, q_values, %{
        action_counts: action_counts,
        total_steps: 20,
        c: 2.0
      })
      
      assert action == :action3
    end

    test "balances Q-values and confidence" do
      q_values = %{:action1 => 0.9, :action2 => 0.1}
      action_counts = %{:action1 => 100, :action2 => 1}  # action2 has high confidence
      
      action = Policy.select_action(:ucb, q_values, %{
        action_counts: action_counts,
        total_steps: 101,
        c: 2.0
      })
      
      # With low visit count, action2 might be selected despite lower Q-value
      assert action in [:action1, :action2]
    end

    test "works with tensor Q-values" do
      q_values = Nx.tensor([0.5, 0.8, 0.2])
      action_counts = %{0 => 5, 1 => 10}  # Action 2 unvisited
      
      action = Policy.select_action(:ucb, q_values, %{
        action_counts: action_counts,
        total_steps: 20
      })
      
      assert action == 2  # Should select unvisited action
    end
  end

  describe "epsilon schedules" do
    test "linear schedule decreases epsilon over time" do
      options = %{start_epsilon: 1.0, end_epsilon: 0.1, decay_steps: 100}
      
      epsilon_0 = Policy.epsilon_schedule(0, :linear, options)
      epsilon_50 = Policy.epsilon_schedule(50, :linear, options)
      epsilon_100 = Policy.epsilon_schedule(100, :linear, options)
      
      assert epsilon_0 == 1.0
      assert_in_delta epsilon_50, 0.55, 0.001
      assert epsilon_100 == 0.1
    end

    test "linear schedule doesn't go below minimum" do
      options = %{start_epsilon: 1.0, end_epsilon: 0.1, decay_steps: 100}
      
      epsilon_200 = Policy.epsilon_schedule(200, :linear, options)
      assert epsilon_200 == 0.1
    end

    test "exponential schedule decays exponentially" do
      options = %{start_epsilon: 1.0, end_epsilon: 0.01, decay_rate: 0.95}
      
      epsilon_0 = Policy.epsilon_schedule(0, :exponential, options)
      epsilon_10 = Policy.epsilon_schedule(10, :exponential, options)
      
      assert epsilon_0 == 1.0
      assert epsilon_10 < epsilon_0
      assert epsilon_10 >= 0.01
    end

    test "cosine schedule follows cosine curve" do
      options = %{start_epsilon: 1.0, end_epsilon: 0.0, decay_steps: 100}
      
      epsilon_0 = Policy.epsilon_schedule(0, :cosine, options)
      epsilon_50 = Policy.epsilon_schedule(50, :cosine, options)
      epsilon_100 = Policy.epsilon_schedule(100, :cosine, options)
      
      assert epsilon_0 == 1.0
      assert epsilon_50 == 0.5
      assert_in_delta epsilon_100, 0.0, 0.001
    end
  end

  describe "adaptive epsilon" do
    test "increases epsilon when performance is poor" do
      current_epsilon = 0.1
      recent_rewards = [1.0, 2.0, 1.5]  # Average: 1.5
      target_reward = 5.0
      
      new_epsilon = Policy.adaptive_epsilon(current_epsilon, recent_rewards, target_reward)
      assert new_epsilon > current_epsilon
    end

    test "decreases epsilon when performance is good" do
      current_epsilon = 0.5
      recent_rewards = [8.0, 9.0, 10.0]  # Average: 9.0
      target_reward = 5.0
      
      new_epsilon = Policy.adaptive_epsilon(current_epsilon, recent_rewards, target_reward)
      assert new_epsilon < current_epsilon
    end

    test "respects minimum and maximum bounds" do
      options = %{min_epsilon: 0.01, max_epsilon: 0.9}
      
      # Test minimum bound
      low_epsilon = Policy.adaptive_epsilon(0.02, [10.0], 5.0, options)
      assert low_epsilon >= 0.01
      
      # Test maximum bound  
      high_epsilon = Policy.adaptive_epsilon(0.8, [1.0], 10.0, options)
      assert high_epsilon <= 0.9
    end
  end

  describe "EpsilonGreedy module" do
    test "direct module usage with maps" do
      q_values = %{:a => 0.5, :b => 0.8, :c => 0.2}
      
      action = Policy.EpsilonGreedy.select_action(q_values, 0.0)
      assert action == :b
      
      # Test exploration
      actions = Enum.map(1..50, fn _ ->
        Policy.EpsilonGreedy.select_action(q_values, 1.0)
      end)
      assert length(Enum.uniq(actions)) > 1
    end

    test "direct module usage with tensors" do
      q_values = Nx.tensor([0.2, 0.8, 0.5])
      
      action = Policy.EpsilonGreedy.select_action(q_values, 0.0)
      assert action == 1
    end
  end

  describe "Greedy module" do
    test "direct module usage" do
      q_values = %{:x => 0.3, :y => 0.7, :z => 0.1}
      action = Policy.Greedy.select_action(q_values)
      assert action == :y
    end
  end

  describe "Boltzmann module" do
    test "direct module usage with maps" do
      q_values = %{:a => 1.0, :b => 2.0}
      action = Policy.Boltzmann.select_action(q_values, 1.0)
      assert action in [:a, :b]
    end

    test "direct module usage with tensors" do
      q_values = Nx.tensor([1.0, 2.0])
      action = Policy.Boltzmann.select_action(q_values, 1.0)
      assert action in [0, 1]
    end
  end

  describe "UpperConfidenceBound module" do
    test "direct module usage with maps" do
      q_values = %{:a => 0.5, :b => 0.6}
      action_counts = %{:a => 10, :b => 5}
      
      action = Policy.UpperConfidenceBound.select_action(q_values, action_counts, 20)
      assert action in [:a, :b]
    end

    test "direct module usage with tensors" do
      q_values = Nx.tensor([0.5, 0.6])
      action_counts = %{0 => 10, 1 => 5}
      
      action = Policy.UpperConfidenceBound.select_action(q_values, action_counts, 20)
      assert action in [0, 1]
    end
  end
end