defmodule QLearning.Environments.CartPoleTest do
  use ExUnit.Case
  alias QLearning.Environments.CartPole

  describe "Environment behaviour implementation" do
    test "get_actions returns left and right actions" do
      actions = CartPole.get_actions()
      assert actions == [0, 1]
    end

    test "reset returns initial state as list" do
      initial_state = CartPole.reset()
      assert initial_state == [0.0, 0.0, 0.0, 0.0]
    end

    test "get_states returns empty list for continuous space" do
      states = CartPole.get_states()
      assert states == []
    end
  end

  describe "stateless step function" do
    test "step takes action and returns next state, reward, done" do
      initial_state = [0.0, 0.0, 0.0, 0.0]
      {next_state, reward, done} = CartPole.step(initial_state, 1)
      
      assert is_list(next_state)
      assert length(next_state) == 4
      assert is_number(reward)
      assert is_boolean(done)
    end

    test "step with right action (1) applies positive force" do
      initial_state = [0.0, 0.0, 0.0, 0.0]
      {next_state, _reward, _done} = CartPole.step(initial_state, 1)
      
      [cart_pos, cart_vel, _pole_angle, _pole_vel] = next_state
      assert cart_vel > 0  # Cart should move right
    end

    test "step with left action (0) applies negative force" do
      initial_state = [0.0, 0.0, 0.0, 0.0]
      {next_state, _reward, _done} = CartPole.step(initial_state, 0)
      
      [cart_pos, cart_vel, _pole_angle, _pole_vel] = next_state
      assert cart_vel < 0  # Cart should move left
    end

    test "episode ends when cart moves too far" do
      # Start with cart beyond boundary
      initial_state = [2.5, 0.0, 0.0, 0.0]  # Beyond threshold of 2.4
      {_next_state, _reward, done} = CartPole.step(initial_state, 1)
      
      assert done == true
    end

    test "episode ends when pole angle is too large" do
      # Start with pole at large angle (in radians)
      large_angle = 0.25  # > 12 degrees in radians (~0.209)
      initial_state = [0.0, 0.0, large_angle, 0.0]
      {_next_state, _reward, done} = CartPole.step(initial_state, 1)
      
      assert done == true
    end
  end

  describe "GenServer-based CartPole" do
    test "can start and stop CartPole process" do
      {:ok, pid} = CartPole.start_link()
      assert Process.alive?(pid)
      
      GenServer.stop(pid)
      refute Process.alive?(pid)
    end

    test "reset returns initial state vector" do
      {:ok, pid} = CartPole.start_link()
      
      state = CartPole.reset(pid)
      assert is_list(state)
      assert length(state) == 4
      assert Enum.all?(state, &is_number/1)
      
      GenServer.stop(pid)
    end

    test "get_state returns current state vector" do
      {:ok, pid} = CartPole.start_link()
      
      state = CartPole.get_state(pid)
      assert is_list(state)
      assert length(state) == 4
      
      GenServer.stop(pid)
    end

    test "step updates state and returns transition" do
      {:ok, pid} = CartPole.start_link()
      
      initial_state = CartPole.get_state(pid)
      {next_state, reward, done} = CartPole.step(pid, 1)
      
      assert next_state != initial_state
      assert is_number(reward)
      assert is_boolean(done)
      
      GenServer.stop(pid)
    end

    test "multiple steps increment step count" do
      {:ok, pid} = CartPole.start_link(max_steps: 5)
      
      # Take several steps
      Enum.each(1..3, fn _ ->
        CartPole.step(pid, 1)
      end)
      
      # Take a step that should be close to max_steps
      {_state, _reward, done} = CartPole.step(pid, 1)
      
      GenServer.stop(pid)
    end
  end

  describe "utility functions" do
    test "normalize_state scales values appropriately" do
      state = [1.0, 0.5, 0.1, 0.2]
      normalized = CartPole.normalize_state(state)
      
      assert is_list(normalized)
      assert length(normalized) == 4
      assert Enum.all?(normalized, &is_number/1)
      
      # First element should be normalized by x_threshold (2.4)
      assert_in_delta List.first(normalized), 1.0 / 2.4, 0.001
    end

    test "observation_space_size returns 4" do
      assert CartPole.observation_space_size() == 4
    end

    test "action_space_size returns 2" do
      assert CartPole.action_space_size() == 2
    end

    test "create_stateless_env returns module name" do
      env = CartPole.create_stateless_env()
      assert env == QLearning.Environments.CartPole
    end
  end

  describe "physics simulation" do
    test "state values change after step" do
      initial_state = [0.0, 0.0, 0.0, 0.0]
      {next_state, _reward, _done} = CartPole.step(initial_state, 1)
      
      [cart_pos, cart_vel, pole_angle, pole_vel] = next_state
      
      # At least some values should change due to physics
      assert cart_pos != 0.0 or cart_vel != 0.0 or pole_angle != 0.0 or pole_vel != 0.0
    end

    test "consecutive steps show progression" do
      state1 = [0.0, 0.0, 0.0, 0.0]
      {state2, _reward, _done} = CartPole.step(state1, 1)
      {state3, _reward, _done} = CartPole.step(state2, 1)
      
      # Cart velocity should increase with continued force
      [_pos1, vel1, _angle1, _avel1] = state2
      [_pos2, vel2, _angle2, _avel2] = state3
      
      assert vel2 > vel1  # Velocity should increase
    end
  end
end