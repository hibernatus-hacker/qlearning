defmodule QLearning.MixProject do
  use Mix.Project

  def project do
    [
      app: :qlearning,
      version: "0.1.0",
      elixir: "~> 1.14",
      start_permanent: Mix.env() == :prod,
      deps: deps(),
      description: description(),
      package: package(),
      docs: docs()
    ]
  end

  def application do
    [
      extra_applications: [:logger]
    ]
  end

  defp deps do
    [
      {:nx, "~> 0.6"},
      {:axon, "~> 0.6"},
      {:jason, "~> 1.4"},
      {:ex_doc, "~> 0.31", only: :dev, runtime: false},
      {:dialyxir, "~> 1.3", only: [:dev], runtime: false}
    ]
  end

  defp description do
    "A comprehensive Q-learning reinforcement learning library for Elixir"
  end

  defp package do
    [
      licenses: ["MIT"],
      links: %{"GitHub" => "https://github.com/your-username/qlearning"}
    ]
  end

  defp docs do
    [
      main: "QLearning",
      extras: ["README.md"]
    ]
  end
end