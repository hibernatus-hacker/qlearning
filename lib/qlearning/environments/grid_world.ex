defmodule QLearning.Environments.GridWorld do
  @moduledoc """
  A simple GridWorld environment for testing Q-learning algorithms.
  
  The agent navigates a grid to reach a goal while avoiding obstacles.
  - States: Grid positions {x, y}
  - Actions: :up, :down, :left, :right
  - Rewards: +10 for goal, -1 for each step, -10 for hitting walls/obstacles
  """

  @behaviour QLearning.Environment

  defstruct [:grid, :agent_pos, :goal_pos, :obstacles, :width, :height]

  @type position :: {integer(), integer()}
  @type grid_state :: %__MODULE__{
    grid: list(list(atom())),
    agent_pos: position(),
    goal_pos: position(),
    obstacles: [position()],
    width: integer(),
    height: integer()
  }

  @actions [:up, :down, :left, :right]

  @doc """
  Creates a new GridWorld environment.
  
  ## Parameters
  - `width`: Grid width
  - `height`: Grid height  
  - `goal_pos`: Goal position {x, y}
  - `obstacles`: List of obstacle positions
  - `start_pos`: Starting position (default: {0, 0})
  """
  def new(width, height, goal_pos, obstacles \\ [], start_pos \\ {0, 0}) do
    grid = create_grid(width, height, goal_pos, obstacles)
    
    %__MODULE__{
      grid: grid,
      agent_pos: start_pos,
      goal_pos: goal_pos,
      obstacles: obstacles,
      width: width,
      height: height
    }
  end

  @doc """
  Creates a simple 4x4 GridWorld for quick testing.
  """
  def simple_4x4() do
    new(4, 4, {3, 3}, [{1, 1}, {2, 2}], {0, 0})
  end

  # Environment behaviour implementation

  @impl true
  def get_states() do
    # For a generic GridWorld, we'll generate all possible positions
    # In practice, this would be customized per specific grid
    for x <- 0..9, y <- 0..9, do: {x, y}
  end

  @impl true
  def get_actions(), do: @actions

  @impl true
  def reset() do
    simple_4x4().agent_pos
  end

  @impl true
  def step(state, action) do
    grid_world = simple_4x4()
    current_pos = state
    
    new_pos = move(current_pos, action)
    {reward, done} = calculate_reward(grid_world, current_pos, new_pos)
    
    # If move is invalid, stay in same position
    final_pos = if valid_position?(grid_world, new_pos), do: new_pos, else: current_pos
    
    {final_pos, reward, done}
  end

  @impl true  
  def get_valid_actions(state) do
    grid_world = simple_4x4()
    
    @actions
    |> Enum.filter(fn action ->
      new_pos = move(state, action)
      valid_position?(grid_world, new_pos)
    end)
  end

  # Stateful version for DQN training

  @doc """
  Stateful GridWorld for use with DQN agents.
  """
  def start_link(config \\ %{}) do
    GenServer.start_link(__MODULE__.Stateful, config)
  end

  defmodule Stateful do
    use GenServer
    alias QLearning.Environments.GridWorld
    
    @actions [:up, :down, :left, :right]

    def init(config) do
      width = Map.get(config, :width, 4)
      height = Map.get(config, :height, 4)
      goal_pos = Map.get(config, :goal_pos, {3, 3})
      obstacles = Map.get(config, :obstacles, [{1, 1}, {2, 2}])
      start_pos = Map.get(config, :start_pos, {0, 0})
      
      state = GridWorld.new(width, height, goal_pos, obstacles, start_pos)
      {:ok, state}
    end

    def handle_call(:reset, _from, grid_world) do
      new_state = %{grid_world | agent_pos: {0, 0}}
      state_vector = position_to_vector(new_state.agent_pos, new_state.width, new_state.height)
      {:reply, state_vector, new_state}
    end

    def handle_call(:get_states, _from, grid_world) do
      states = for x <- 0..(grid_world.width-1), y <- 0..(grid_world.height-1) do
        position_to_vector({x, y}, grid_world.width, grid_world.height)
      end
      {:reply, states, grid_world}
    end

    def handle_call(:get_actions, _from, grid_world) do
      {:reply, Enum.with_index(@actions), grid_world}
    end

    def handle_call({:step, action_index}, _from, grid_world) do
      action = Enum.at(@actions, action_index)
      current_pos = grid_world.agent_pos
      
      new_pos = GridWorld.move(current_pos, action)
      {reward, done} = GridWorld.calculate_reward(grid_world, current_pos, new_pos)
      
      final_pos = if GridWorld.valid_position?(grid_world, new_pos), do: new_pos, else: current_pos
      
      new_grid_world = %{grid_world | agent_pos: final_pos}
      next_state_vector = position_to_vector(final_pos, grid_world.width, grid_world.height)
      
      {:reply, {next_state_vector, reward, done}, new_grid_world}
    end

    def handle_call({:get_valid_actions, _state}, _from, grid_world) do
      valid_actions = 
        @actions
        |> Enum.with_index()
        |> Enum.filter(fn {action, _idx} ->
          new_pos = GridWorld.move(grid_world.agent_pos, action)
          GridWorld.valid_position?(grid_world, new_pos)
        end)
        |> Enum.map(fn {_action, idx} -> idx end)
      
      {:reply, valid_actions, grid_world}
    end

    defp position_to_vector({x, y}, width, height) do
      # One-hot encoding of position
      vector = List.duplicate(0.0, width * height)
      index = y * width + x
      List.replace_at(vector, index, 1.0)
    end
  end

  # Helper functions

  def move({x, y}, :up), do: {x, y - 1}
  def move({x, y}, :down), do: {x, y + 1}
  def move({x, y}, :left), do: {x - 1, y}
  def move({x, y}, :right), do: {x + 1, y}

  def valid_position?(%__MODULE__{} = grid_world, {x, y}) do
    x >= 0 and x < grid_world.width and 
    y >= 0 and y < grid_world.height and
    {x, y} not in grid_world.obstacles
  end

  def calculate_reward(%__MODULE__{} = grid_world, current_pos, new_pos) do
    cond do
      new_pos == grid_world.goal_pos ->
        {10.0, true}  # Reached goal
        
      not valid_position?(grid_world, new_pos) ->
        {-10.0, false}  # Hit wall or obstacle
        
      true ->
        {-1.0, false}  # Normal step cost
    end
  end

  defp create_grid(width, height, goal_pos, obstacles) do
    for y <- 0..(height-1) do
      for x <- 0..(width-1) do
        cond do
          {x, y} == goal_pos -> :goal
          {x, y} in obstacles -> :obstacle
          true -> :empty
        end
      end
    end
  end

  @doc """
  Prints a visual representation of the grid world.
  """
  def print_grid(%__MODULE__{} = grid_world) do
    IO.puts("\nGridWorld (A=Agent, G=Goal, X=Obstacle, .=Empty):")
    
    for {row, y} <- Enum.with_index(grid_world.grid) do
      row_str = 
        for {cell, x} <- Enum.with_index(row) do
          cond do
            {x, y} == grid_world.agent_pos -> "A"
            cell == :goal -> "G"
            cell == :obstacle -> "X"
            true -> "."
          end
        end
        |> Enum.join(" ")
      
      IO.puts(row_str)
    end
    
    IO.puts("")
  end
end