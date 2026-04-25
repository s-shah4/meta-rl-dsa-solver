import torch
from unsloth import FastLanguageModel, PatchFastRL
from trl import GRPOTrainer, GRPOConfig
from meta_rl_dsa_solver_env import DsaEnv

# 1. Patch Unsloth for RL speedups
PatchFastRL("GRPO", FastLanguageModel)

# 2. Load Model & Tokenizer
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Llama-3.2-3B-Instruct", # Use appropriate 2026 base
    max_seq_length = 2048,
    load_in_4bit = True,
    fast_inference = True,
)

model = FastLanguageModel.get_peft_model(
    model,
    r = 16,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_alpha = 16,
    lora_dropout = 0,
)

# 3. Define the Reward Function (Interface for Person 2)
def reward_function(prompts, completions, **kwargs) -> list[float]:
    """
    In GRPO, the reward function is called on the batch of completions.
    For V0, we manually trigger the Env's step logic.
    """
    env = DsaEnv()
    rewards = []
    
    for completion in completions:
        # Extract code from completion (assuming markdown tags)
        code = completion.split("```python")[-1].split("```")[0].strip() if "```" in completion else completion
        _, reward, _, _, _ = env.step(code)
        rewards.append(reward)
    
    return rewards

# 4. Training Configuration
training_args = GRPOConfig(
    output_dir = "./outputs",
    learning_rate = 5e-6,
    per_device_train_batch_size = 4,
    gradient_accumulation_steps = 4,
    max_prompt_length = 512,
    max_completion_length = 512,
    num_generations = 8, # Group size for GRPO
    logging_steps = 1,
    max_steps = 100, # Quick run for MVP
)

# 5. Initialize Trainer
trainer = GRPOTrainer(
    model = model,
    reward_funcs = [reward_function],
    args = training_args,
    train_dataset = [
        {"prompt": "Write a function `sum_list(arr: list) -> int` that returns the sum of a list."}
    ] * 100, # Dummy dataset for V0 validation
)

if __name__ == "__main__":
    print("Starting V0 Training...")
    trainer.train()
    model.save_pretrained_merged("final_v0_model", tokenizer, save_method = "merged_16bit")