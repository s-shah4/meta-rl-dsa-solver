from meta_rl_dsa_solver_env.env import DsaEnv

env = DsaEnv()
obs, info = env.reset()
print(f"Observation: {obs}")

# Simulate a correct LLM response
sample_code = "def sum_list(arr):\n    return sum(arr)"
obs, reward, terminated, truncated, info = env.step(sample_code)

print(f"Reward: {reward}") # Should be 1.0
print(f"Success: {info['success']}")