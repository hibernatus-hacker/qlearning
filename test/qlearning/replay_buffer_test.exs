defmodule QLearning.ReplayBufferTest do
  use ExUnit.Case
  alias QLearning.ReplayBuffer

  describe "ReplayBuffer initialization" do
    test "can start and stop buffer process" do
      {:ok, pid} = ReplayBuffer.start_link(100, {4})
      assert Process.alive?(pid)
      
      GenServer.stop(pid)
      refute Process.alive?(pid)
    end

    test "initializes with correct capacity and state shape" do
      {:ok, pid} = ReplayBuffer.start_link(50, {2})
      
      assert ReplayBuffer.size(pid) == 0
      assert ReplayBuffer.capacity(pid) == 50
      
      GenServer.stop(pid)
    end
  end

  describe "adding transitions" do
    setup do
      {:ok, pid} = ReplayBuffer.start_link(10, {2})
      
      transition = %{
        state: Nx.tensor([1.0, 2.0]),
        action: 0,
        reward: 1.0,
        next_state: Nx.tensor([3.0, 4.0]),
        done: false
      }
      
      %{buffer: pid, transition: transition}
    end

    test "can add single transition", %{buffer: buffer, transition: transition} do
      ReplayBuffer.add(buffer, transition)
      assert ReplayBuffer.size(buffer) == 1
      GenServer.stop(buffer)
    end

    test "can add multiple transitions", %{buffer: buffer, transition: transition} do
      Enum.each(1..5, fn _ ->
        ReplayBuffer.add(buffer, transition)
      end)
      
      assert ReplayBuffer.size(buffer) == 5
      GenServer.stop(buffer)
    end

    test "overwrites old transitions when capacity exceeded", %{buffer: buffer, transition: transition} do
      # Add more transitions than capacity
      Enum.each(1..15, fn _ ->
        ReplayBuffer.add(buffer, transition)
      end)
      
      # Size should not exceed capacity
      assert ReplayBuffer.size(buffer) == 10
      GenServer.stop(buffer)
    end
  end

  describe "sampling transitions" do
    setup do
      {:ok, buffer} = ReplayBuffer.start_link(100, {2})
      
      # Add several different transitions
      transitions = [
        %{state: Nx.tensor([1.0, 1.0]), action: 0, reward: 1.0, next_state: Nx.tensor([2.0, 2.0]), done: false},
        %{state: Nx.tensor([2.0, 2.0]), action: 1, reward: 2.0, next_state: Nx.tensor([3.0, 3.0]), done: false},
        %{state: Nx.tensor([3.0, 3.0]), action: 0, reward: 3.0, next_state: Nx.tensor([4.0, 4.0]), done: true},
        %{state: Nx.tensor([4.0, 4.0]), action: 1, reward: 4.0, next_state: Nx.tensor([5.0, 5.0]), done: false}
      ]
      
      Enum.each(transitions, &ReplayBuffer.add(buffer, &1))
      
      %{buffer: buffer, transitions: transitions}
    end

    test "can sample batch when buffer has enough transitions", %{buffer: buffer} do
      batch = ReplayBuffer.sample(buffer, 2)
      
      assert batch != nil
      assert Map.has_key?(batch, :states)
      assert Map.has_key?(batch, :actions) 
      assert Map.has_key?(batch, :rewards)
      assert Map.has_key?(batch, :next_states)
      assert Map.has_key?(batch, :dones)
      
      GenServer.stop(buffer)
    end

    test "returns nil when trying to sample more than available", %{buffer: buffer} do
      batch = ReplayBuffer.sample(buffer, 10)  # More than 4 transitions
      assert batch == nil
      
      GenServer.stop(buffer)
    end

    test "sampled batch has correct dimensions", %{buffer: buffer} do
      batch = ReplayBuffer.sample(buffer, 3)
      
      # States should be batch_size x state_shape
      assert Nx.shape(batch.states) == {3, 2}
      assert Nx.shape(batch.next_states) == {3, 2}
      
      # Actions, rewards, dones should be batch_size
      assert Nx.shape(batch.actions) == {3}
      assert Nx.shape(batch.rewards) == {3}
      assert Nx.shape(batch.dones) == {3}
      
      GenServer.stop(buffer)
    end
  end

  describe "buffer state queries" do
    test "size and capacity work correctly" do
      {:ok, buffer} = ReplayBuffer.start_link(20, {3})
      
      assert ReplayBuffer.size(buffer) == 0
      assert ReplayBuffer.capacity(buffer) == 20
      
      # Add some transitions
      transition = %{
        state: Nx.tensor([1.0, 2.0, 3.0]),
        action: 0,
        reward: 1.0,
        next_state: Nx.tensor([2.0, 3.0, 4.0]),
        done: false
      }
      
      Enum.each(1..5, fn _ -> ReplayBuffer.add(buffer, transition) end)
      
      assert ReplayBuffer.size(buffer) == 5
      assert ReplayBuffer.capacity(buffer) == 20
      
      GenServer.stop(buffer)
    end

    test "is_ready returns correct readiness status" do
      {:ok, buffer} = ReplayBuffer.start_link(10, {2})
      
      # Not ready when empty
      assert ReplayBuffer.is_ready(buffer, 5) == false
      
      # Add some transitions
      transition = %{
        state: Nx.tensor([1.0, 2.0]),
        action: 0,
        reward: 1.0,
        next_state: Nx.tensor([2.0, 3.0]),
        done: false
      }
      
      Enum.each(1..3, fn _ -> ReplayBuffer.add(buffer, transition) end)
      
      # Still not ready if we need more than available
      assert ReplayBuffer.is_ready(buffer, 5) == false
      
      # Ready when we have enough
      assert ReplayBuffer.is_ready(buffer, 3) == true
      assert ReplayBuffer.is_ready(buffer, 2) == true
      
      GenServer.stop(buffer)
    end
  end

  describe "edge cases" do
    test "handles buffer with size 1" do
      {:ok, buffer} = ReplayBuffer.start_link(1, {1})
      
      transition1 = %{
        state: Nx.tensor([1.0]),
        action: 0, 
        reward: 1.0,
        next_state: Nx.tensor([2.0]),
        done: false
      }
      
      transition2 = %{
        state: Nx.tensor([3.0]),
        action: 1,
        reward: 2.0, 
        next_state: Nx.tensor([4.0]),
        done: true
      }
      
      ReplayBuffer.add(buffer, transition1)
      assert ReplayBuffer.size(buffer) == 1
      
      ReplayBuffer.add(buffer, transition2)
      assert ReplayBuffer.size(buffer) == 1  # Should still be 1 due to capacity
      
      GenServer.stop(buffer)
    end

    test "handles empty state shape" do
      # This might not be practical but tests edge case handling
      {:ok, buffer} = ReplayBuffer.start_link(5, {})
      
      transition = %{
        state: Nx.tensor(1.0),  # Scalar tensor
        action: 0,
        reward: 1.0,
        next_state: Nx.tensor(2.0),
        done: false
      }
      
      ReplayBuffer.add(buffer, transition)
      assert ReplayBuffer.size(buffer) == 1
      
      GenServer.stop(buffer)
    end
  end
end