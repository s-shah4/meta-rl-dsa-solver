import torch
import numpy as np
from unsloth import FastLanguageModel, PatchFastRL
from trl import GRPOTrainer, GRPOConfig
from meta_rl_dsa_solver_env.env.adapt_env import AdaptEnvironment
from meta_rl_dsa_solver_env.models import AdaptAction

# 1. Initialize Model & Speedups
PatchFastRL("GRPO", FastLanguageModel)

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Llama-3.2-3B-Instruct",
    max_seq_length = 2048,
    load_in_4bit = True,
)

model = FastLanguageModel.get_peft_model(
    model,
    r = 16,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_alpha = 16,
)

# 2. V2 Heuristic State Machine
class CurriculumManager:
    def __init__(self):
        self.difficulties = ["easy", "medium", "hard"]
        self.current_idx = 0
        self.success_history = []
        self.window_size = 10  # Moving average window

    def get_current_difficulty(self):
        return self.difficulties[self.current_idx]

    def update(self, success_rate):
        self.success_history.append(success_rate)
        if len(self.success_history) > self.window_size:
            self.success_history.pop(0)
        
        # V2 Logic: If moving average > 70%, increase difficulty
        avg_success = np.mean(self.success_history)
        if avg_success > 0.70 and self.current_idx < len(self.difficulties) - 1:
            self.current_idx += 1
            print(f"--- HEURISTIC LEVEL UP: Moving to {self.difficulties[self.current_idx]} ---")
            self.success_history = [] # Reset for the new tier

curriculum = CurriculumManager()

# 3. V2 Reward Function with Curriculum Feedback
def v2_reward_func(prompts, completions, **kwargs) -> list[float]:
    env = AdaptEnvironment()
    rewards = []
    successes = []
    
    current_diff = curriculum.get_current_difficulty()
    
    for completion in completions:
        # Load problem based on current heuristic difficulty
        env.reset(difficulty=current_diff)
        
        code = completion.split("```python")[-1].split("```")[0].strip() if "```" in completion else completion
        action = AdaptAction(code=code)
        obs = env.step(action)
        
        rewards.append(float(obs.reward))
        successes.append(1.0 if obs.pass_rate == 1.0 else 0.0)
    
    # Update the curriculum manager based on this batch
    batch_success_rate = np.mean(successes)
    curriculum.update(batch_success_rate)
    
    return rewards

# 4. Dataset: Transition from single prompt to generic instruction
# This forces the LLM to look at the 'problem statement' in the observation
dataset = [
    {"prompt": "Read the problem statement and constraints carefully. Write a Python solution that reads from stdin and prints to stdout."}
] * 200 # Larger dataset for multi-tier learning

# 5. Config
training_args = GRPOConfig(
    output_dir = "./outputs_v2",
    learning_rate = 5e-6,
    per_device_train_batch_size = 1,
    gradient_accumulation_steps = 8, # Higher for stability during transitions
    num_generations = 8,
    max_steps = 250,
    bf16 = True,
    logging_steps = 1,
)

trainer = GRPOTrainer(
    model = model,
    reward_funcs = [v2_reward_func],
    args = training_args,
    train_dataset = dataset,
)

if __name__ == "__main__":
    print(f"Starting V2 Training. Initial Difficulty: {curriculum.get_current_difficulty()}")
    trainer.train()