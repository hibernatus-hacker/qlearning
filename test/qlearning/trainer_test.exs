defmodule QLearning.TrainerTest do
  use ExUnit.Case
  alias QLearning.Trainer

  describe "Trainer initialization" do
    test "can start and stop trainer process" do
      {:ok, pid} = Trainer.start_link("test_experiment")
      assert Process.alive?(pid)
      
      GenServer.stop(pid)
      refute Process.alive?(pid)
    end

    test "initializes with experiment name and hyperparameters" do
      hyperparams = %{learning_rate: 0.01, gamma: 0.95}
      {:ok, pid} = Trainer.start_link("test_exp", hyperparams)
      
      stats = Trainer.get_stats(pid)
      assert stats.experiment_name == "test_exp"
      assert stats.hyperparams == hyperparams
      assert stats.current_episode == 0
      assert stats.episode_rewards == []
      
      GenServer.stop(pid)
    end
  end

  describe "metrics recording" do
    setup do
      {:ok, trainer} = Trainer.start_link("metrics_test")
      %{trainer: trainer}
    end

    test "records episode metrics", %{trainer: trainer} do
      Trainer.record_episode(trainer, 10.5, 100, 0.1)
      Trainer.record_episode(trainer, 15.2, 150, 0.09)
      
      stats = Trainer.get_stats(trainer)
      assert length(stats.episode_rewards) == 2
      assert stats.episode_rewards == [10.5, 15.2]
      assert stats.episode_lengths == [100, 150]
      assert stats.epsilon_history == [0.1, 0.09]
      
      GenServer.stop(trainer)
    end

    test "records episode without epsilon", %{trainer: trainer} do
      Trainer.record_episode(trainer, 5.0, 50)
      
      stats = Trainer.get_stats(trainer)
      assert stats.episode_rewards == [5.0]
      assert stats.episode_lengths == [50]
      assert stats.epsilon_history == [nil]
      
      GenServer.stop(trainer)
    end

    test "records training loss", %{trainer: trainer} do
      Trainer.record_loss(trainer, 0.5)
      Trainer.record_loss(trainer, 0.3)
      Trainer.record_loss(trainer, 0.2)
      
      stats = Trainer.get_stats(trainer)
      assert stats.loss_history == [0.5, 0.3, 0.2]
      
      GenServer.stop(trainer)
    end
  end

  describe "statistics calculation" do
    setup do
      {:ok, trainer} = Trainer.start_link("stats_test")
      
      # Add some episode data
      Trainer.record_episode(trainer, 10.0, 100, 0.1)
      Trainer.record_episode(trainer, 20.0, 200, 0.08)
      Trainer.record_episode(trainer, 15.0, 150, 0.06)
      
      %{trainer: trainer}
    end

    test "calculates basic statistics", %{trainer: trainer} do
      stats = Trainer.get_stats(trainer)
      
      assert length(stats.episode_rewards) == 3
      assert stats.current_episode == 3
      
      # Check that start_time is set
      assert stats.start_time != nil
      assert DateTime.diff(DateTime.utc_now(), stats.start_time) >= 0
      
      GenServer.stop(trainer)
    end

    test "maintains episode ordering", %{trainer: trainer} do
      stats = Trainer.get_stats(trainer)
      
      assert stats.episode_rewards == [10.0, 20.0, 15.0]
      assert stats.episode_lengths == [100, 200, 150]
      assert stats.epsilon_history == [0.1, 0.08, 0.06]
      
      GenServer.stop(trainer)
    end
  end

  describe "data export" do
    setup do
      {:ok, trainer} = Trainer.start_link("export_test")
      
      # Add some test data
      Trainer.record_episode(trainer, 10.0, 100, 0.1)
      Trainer.record_episode(trainer, 20.0, 200, 0.08)
      Trainer.record_loss(trainer, 0.5)
      Trainer.record_loss(trainer, 0.3)
      
      %{trainer: trainer}
    end

    test "exports data to JSON file", %{trainer: trainer} do
      filename = "/tmp/test_export.json"
      
      # Clean up any existing file
      if File.exists?(filename), do: File.rm!(filename)
      
      result = Trainer.export_data(trainer, filename)
      assert result == :ok
      
      # Verify file was created
      assert File.exists?(filename)
      
      # Verify file content is valid JSON
      {:ok, content} = File.read(filename)
      {:ok, data} = Jason.decode(content)
      
      assert Map.has_key?(data, "experiment_name")
      assert Map.has_key?(data, "episode_rewards")
      assert Map.has_key?(data, "episode_lengths")
      assert Map.has_key?(data, "loss_history")
      
      assert data["experiment_name"] == "export_test"
      assert data["episode_rewards"] == [10.0, 20.0]
      assert data["loss_history"] == [0.5, 0.3]
      
      # Clean up
      File.rm!(filename)
      GenServer.stop(trainer)
    end

    test "handles export errors gracefully", %{trainer: trainer} do
      invalid_path = "/nonexistent/directory/file.json"
      
      result = Trainer.export_data(trainer, invalid_path)
      assert {:error, _reason} = result
      
      GenServer.stop(trainer)
    end
  end

  describe "concurrent operations" do
    test "handles multiple concurrent recordings" do
      {:ok, trainer} = Trainer.start_link("concurrent_test")
      
      # Simulate concurrent episode recordings
      tasks = Enum.map(1..10, fn i ->
        Task.async(fn ->
          Trainer.record_episode(trainer, Float.round(i * 1.5, 1), i * 10, i * 0.01)
        end)
      end)
      
      Enum.each(tasks, &Task.await/1)
      
      stats = Trainer.get_stats(trainer)
      assert length(stats.episode_rewards) == 10
      assert stats.current_episode == 10
      
      GenServer.stop(trainer)
    end

    test "handles concurrent loss recordings" do
      {:ok, trainer} = Trainer.start_link("loss_concurrent_test")
      
      tasks = Enum.map(1..5, fn i ->
        Task.async(fn ->
          Trainer.record_loss(trainer, i * 0.1)
        end)
      end)
      
      Enum.each(tasks, &Task.await/1)
      
      stats = Trainer.get_stats(trainer)
      assert length(stats.loss_history) == 5
      
      GenServer.stop(trainer)
    end
  end

  describe "edge cases" do
    test "handles zero and negative rewards" do
      {:ok, trainer} = Trainer.start_link("edge_case_test")
      
      Trainer.record_episode(trainer, 0.0, 50)
      Trainer.record_episode(trainer, -10.5, 25)
      
      stats = Trainer.get_stats(trainer)
      assert stats.episode_rewards == [0.0, -10.5]
      assert stats.episode_lengths == [50, 25]
      
      GenServer.stop(trainer)
    end

    test "handles large numbers" do
      {:ok, trainer} = Trainer.start_link("large_numbers_test")
      
      large_reward = 1_000_000.0
      large_length = 50_000
      
      Trainer.record_episode(trainer, large_reward, large_length)
      
      stats = Trainer.get_stats(trainer)
      assert stats.episode_rewards == [large_reward]
      assert stats.episode_lengths == [large_length]
      
      GenServer.stop(trainer)
    end

    test "handles empty experiment name" do
      {:ok, trainer} = Trainer.start_link("")
      
      stats = Trainer.get_stats(trainer)
      assert stats.experiment_name == ""
      
      GenServer.stop(trainer)
    end
  end

  describe "memory management" do
    test "doesn't accumulate unlimited data" do
      {:ok, trainer} = Trainer.start_link("memory_test")
      
      # Add a large number of episodes
      Enum.each(1..1000, fn i ->
        Trainer.record_episode(trainer, Float.round(i * 0.1, 1), i)
      end)
      
      stats = Trainer.get_stats(trainer)
      assert length(stats.episode_rewards) == 1000
      assert stats.current_episode == 1000
      
      # Verify the data is still accessible
      assert List.first(stats.episode_rewards) == 0.1
      assert List.last(stats.episode_rewards) == 100.0
      
      GenServer.stop(trainer)
    end
  end
end