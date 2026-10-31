# Running the original CoDrivingLLM

The CoDrivingLLM repository (Fang et al., 2025,
https://github.com/FanGShiYuu/CoDrivingLLM) is published **without a licence**, so
none of its code is copied into CAVBench. To compare against the original
implementation, add it as a git submodule and run it in its own environment:

```bash
git submodule add https://github.com/FanGShiYuu/CoDrivingLLM third_party/CoDrivingLLM
cd third_party/CoDrivingLLM
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt        # gym 0.15, openai 1.2.2, ... (old versions)
export OPENAI_API_KEY=...
python Run_multi_CAV_LLM.py
```

To inject the same perception noise as CAVBench, perturb the vehicle positions and
speeds that its negotiation and action modules read (in
`llm_controller/llm_agent_negotiation_system.py`, where conflicts and distances to
the conflict point are computed) with Gaussian noise of the levels in
`cavbench/noise/gaussian.py`, keeping the true states for the simulator itself.

Inside CAVBench, the arbiter `codrivingllm` (`cavbench/arbiters/codrivingllm_wrapper.py`)
provides an LLM arbiter written from the paper's description, which runs in the
same environment, on the same noisy states and matched seeds as the other methods.
