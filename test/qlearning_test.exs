defmodule QLearningTest do
  use ExUnit.Case
  doctest QLearning

  describe "init_q_table/2" do
    test "creates Q-table with all zeros for given states and actions" do
      states = [:s1, :s2]
      actions = [:up, :down]
      
      q_table = QLearning.init_q_table(states, actions)
      
      assert q_table == %{
        {:s1, :up} => 0.0,
        {:s1, :down} => 0.0,
        {:s2, :up} => 0.0,
        {:s2, :down} => 0.0
      }
    end

    test "handles empty states and actions" do
      assert QLearning.init_q_table([], []) == %{}
      assert QLearning.init_q_table([:s1], []) == %{}
      assert QLearning.init_q_table([], [:up]) == %{}
    end

    test "handles single state and action" do
      q_table = QLearning.init_q_table([:state], [:action])
      assert q_table == %{{:state, :action} => 0.0}
    end
  end

  describe "choose_action/4" do
    setup do
      q_table = %{
        {:s1, :up} => 0.5,
        {:s1, :down} => 0.8,
        {:s1, :left} => 0.2
      }
      actions = [:up, :down, :left]
      
      %{q_table: q_table, actions: actions}
    end

    test "chooses best action when epsilon is 0 (pure exploitation)", %{q_table: q_table, actions: actions} do
      action = QLearning.choose_action(:s1, q_table, actions, 0.0)
      assert action == :down
    end

    test "chooses random action when epsilon is 1 (pure exploration)", %{q_table: q_table, actions: actions} do
      action = QLearning.choose_action(:s1, q_table, actions, 1.0)
      assert action in actions
    end

    test "handles unknown state by using default Q-values", %{actions: actions} do
      q_table = %{}
      action = QLearning.choose_action(:unknown_state, q_table, actions, 0.0)
      assert action in actions
    end

    test "handles single action", %{q_table: q_table} do
      action = QLearning.choose_action(:s1, q_table, [:up], 0.0)
      assert action == :up
    end
  end

  describe "update_q_value/8" do
    test "updates Q-value using Q-learning formula" do
      q_table = %{
        {:s1, :up} => 0.0,
        {:s2, :up} => 0.5,
        {:s2, :down} => 0.3
      }
      actions = [:up, :down]
      
      new_q_table = QLearning.update_q_value(q_table, :s1, :up, 1.0, :s2, actions, 0.1, 0.9)
      
      # Expected: 0.0 + 0.1 * (1.0 + 0.9 * 0.5 - 0.0) = 0.145
      assert_in_delta new_q_table[{:s1, :up}], 0.145, 0.001
      assert new_q_table[{:s2, :up}] == 0.5
      assert new_q_table[{:s2, :down}] == 0.3
    end

    test "handles terminal state (no next state Q-value)" do
      q_table = %{{:s1, :up} => 0.0}
      actions = [:up]
      
      new_q_table = QLearning.update_q_value(q_table, :s1, :up, 1.0, :terminal, actions, 0.1, 0.9)
      
      # Expected: 0.0 + 0.1 * (1.0 + 0.9 * 0.0 - 0.0) = 0.1
      assert_in_delta new_q_table[{:s1, :up}], 0.1, 0.001
    end

    test "uses default alpha and gamma when not provided" do
      q_table = %{{:s1, :up} => 0.0, {:s2, :up} => 0.5}
      actions = [:up]
      
      new_q_table = QLearning.update_q_value(q_table, :s1, :up, 1.0, :s2, actions)
      
      # Expected with alpha=0.1, gamma=0.9: 0.0 + 0.1 * (1.0 + 0.9 * 0.5 - 0.0) = 0.145
      assert_in_delta new_q_table[{:s1, :up}], 0.145, 0.001
    end

    test "handles new state-action pair" do
      q_table = %{}
      actions = [:up]
      
      new_q_table = QLearning.update_q_value(q_table, :s1, :up, 1.0, :s2, actions, 0.1, 0.9)
      
      # Expected: 0.0 + 0.1 * (1.0 + 0.9 * 0.0 - 0.0) = 0.1
      assert_in_delta new_q_table[{:s1, :up}], 0.1, 0.001
    end
  end

  describe "hyperparameters validation" do
    test "alpha should be between 0 and 1" do
      q_table = %{{:s1, :up} => 0.0}
      actions = [:up]
      
      # Test edge cases
      new_q_table = QLearning.update_q_value(q_table, :s1, :up, 1.0, :s2, actions, 0.0, 0.9)
      assert new_q_table[{:s1, :up}] == 0.0
      
      new_q_table = QLearning.update_q_value(q_table, :s1, :up, 1.0, :s2, actions, 1.0, 0.9)
      assert_in_delta new_q_table[{:s1, :up}], 1.0, 0.001
    end

    test "gamma should be between 0 and 1" do
      q_table = %{{:s1, :up} => 0.0, {:s2, :up} => 0.5}
      actions = [:up]
      
      # Test with gamma = 0 (no future reward consideration)
      new_q_table = QLearning.update_q_value(q_table, :s1, :up, 1.0, :s2, actions, 0.1, 0.0)
      assert_in_delta new_q_table[{:s1, :up}], 0.1, 0.001
      
      # Test with gamma = 1 (full future reward consideration)
      new_q_table = QLearning.update_q_value(q_table, :s1, :up, 1.0, :s2, actions, 0.1, 1.0)
      assert_in_delta new_q_table[{:s1, :up}], 0.15, 0.001
    end
  end
end