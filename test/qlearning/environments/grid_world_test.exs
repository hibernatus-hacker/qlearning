defmodule QLearning.Environments.GridWorldTest do
  use ExUnit.Case
  alias QLearning.Environments.GridWorld

  describe "GridWorld creation" do
    test "creates a GridWorld with specified parameters" do
      grid_world = GridWorld.new(3, 3, {2, 2}, [{1, 1}], {0, 0})
      
      assert grid_world.width == 3
      assert grid_world.height == 3
      assert grid_world.goal_pos == {2, 2}
      assert grid_world.obstacles == [{1, 1}]
      assert grid_world.agent_pos == {0, 0}
    end

    test "creates simple 4x4 GridWorld" do
      grid_world = GridWorld.simple_4x4()
      
      assert grid_world.width == 4
      assert grid_world.height == 4
      assert grid_world.goal_pos == {3, 3}
      assert grid_world.obstacles == [{1, 1}, {2, 2}]
      assert grid_world.agent_pos == {0, 0}
    end
  end

  describe "Environment behaviour implementation" do
    test "get_actions returns expected actions" do
      actions = GridWorld.get_actions()
      assert actions == [:up, :down, :left, :right]
    end

    test "reset returns starting position" do
      initial_state = GridWorld.reset()
      assert initial_state == {0, 0}
    end

    test "get_states returns grid positions" do
      states = GridWorld.get_states()
      assert is_list(states)
      assert {0, 0} in states
      assert {9, 9} in states
    end
  end

  describe "movement mechanics" do
    test "move function calculates new positions correctly" do
      assert GridWorld.move({1, 1}, :up) == {1, 0}
      assert GridWorld.move({1, 1}, :down) == {1, 2}
      assert GridWorld.move({1, 1}, :left) == {0, 1}
      assert GridWorld.move({1, 1}, :right) == {2, 1}
    end
  end

  describe "position validation" do
    setup do
      grid_world = GridWorld.new(3, 3, {2, 2}, [{1, 1}], {0, 0})
      %{grid_world: grid_world}
    end

    test "valid_position? returns true for valid positions", %{grid_world: grid_world} do
      assert GridWorld.valid_position?(grid_world, {0, 0}) == true
      assert GridWorld.valid_position?(grid_world, {2, 2}) == true
      assert GridWorld.valid_position?(grid_world, {0, 2}) == true
    end

    test "valid_position? returns false for positions outside bounds", %{grid_world: grid_world} do
      assert GridWorld.valid_position?(grid_world, {-1, 0}) == false
      assert GridWorld.valid_position?(grid_world, {3, 0}) == false
      assert GridWorld.valid_position?(grid_world, {0, -1}) == false
      assert GridWorld.valid_position?(grid_world, {0, 3}) == false
    end

    test "valid_position? returns false for obstacle positions", %{grid_world: grid_world} do
      assert GridWorld.valid_position?(grid_world, {1, 1}) == false
    end
  end

  describe "reward calculation" do
    setup do
      grid_world = GridWorld.new(3, 3, {2, 2}, [{1, 1}], {0, 0})
      %{grid_world: grid_world}
    end

    test "reaching goal gives positive reward and ends episode", %{grid_world: grid_world} do
      {reward, done} = GridWorld.calculate_reward(grid_world, {2, 1}, {2, 2})
      assert reward == 10.0
      assert done == true
    end

    test "hitting wall gives negative reward", %{grid_world: grid_world} do
      {reward, done} = GridWorld.calculate_reward(grid_world, {0, 0}, {-1, 0})
      assert reward == -10.0
      assert done == false
    end

    test "hitting obstacle gives negative reward", %{grid_world: grid_world} do
      {reward, done} = GridWorld.calculate_reward(grid_world, {1, 0}, {1, 1})
      assert reward == -10.0
      assert done == false
    end

    test "normal step gives small negative reward", %{grid_world: grid_world} do
      {reward, done} = GridWorld.calculate_reward(grid_world, {0, 0}, {0, 1})
      assert reward == -1.0
      assert done == false
    end
  end

  describe "step function" do
    test "step moves agent to valid position" do
      {next_state, reward, done} = GridWorld.step({0, 0}, :right)
      assert next_state == {1, 0}
      assert reward == -1.0
      assert done == false
    end

    test "step keeps agent in same position when hitting wall" do
      {next_state, reward, done} = GridWorld.step({0, 0}, :left)
      assert next_state == {0, 0}
      assert reward == -10.0
      assert done == false
    end

    test "step reaching goal ends episode" do
      {next_state, reward, done} = GridWorld.step({3, 2}, :down)
      assert next_state == {3, 3}
      assert reward == 10.0
      assert done == true
    end
  end

  describe "get_valid_actions" do
    test "returns only valid actions from position" do
      valid_actions = GridWorld.get_valid_actions({0, 0})
      assert :right in valid_actions
      assert :down in valid_actions
      assert :left not in valid_actions
      assert :up not in valid_actions
    end

    test "returns valid actions from position avoiding obstacles" do
      # Position {2, 1} can go up to {2, 0} and right to {3, 1}
      # Cannot go down to {2, 2} (obstacle) or left to {1, 1} (obstacle)
      valid_actions = GridWorld.get_valid_actions({2, 1})
      assert length(valid_actions) == 2
      assert :up in valid_actions
      assert :right in valid_actions
      assert :down not in valid_actions  # Obstacle at {2, 2}
      assert :left not in valid_actions  # Obstacle at {1, 1}
    end

    test "avoids obstacle positions" do
      valid_actions = GridWorld.get_valid_actions({0, 1})
      assert :right not in valid_actions  # Would hit obstacle at {1, 1}
      assert :down in valid_actions
      assert :up in valid_actions
    end
  end
end