import os
import random
from concurrent.futures import ThreadPoolExecutor

import requests
import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM

from tau2.run import get_tasks

from agent import DOMAINS, MODEL, VLLM_URL, rollout

G, TASKS_PER_STEP, EPOCHS = 8, 4, 3          # group size, prompts per step
BETA, KL_COEF, CLIP, MU = 0.2, 0.04, 0.2, 2  # violation penalty, KL weight, PPO clip ε, inner epochs
LR, MINIBATCH = 1e-5, 16
POLICY = os.path.abspath("ckpt/policy")


def token_logprobs(model, prompt, completion):
    ids = torch.tensor(prompt + completion, device=model.device)[None]
    logits = model(ids, logits_to_keep=len(completion) + 1).logits[0, :-1].float()
    return logits.log_softmax(-1).gather(-1, ids[0, -len(completion):, None]).squeeze(-1)


def sync(model):
    model.save_pretrained(POLICY)
    requests.post(f"{VLLM_URL}/v1/load_lora_adapter",
                  json={"lora_name": "policy", "lora_path": POLICY, "load_inplace": True}).raise_for_status()


def main():
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="auto")
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, target_modules="all-linear", task_type="CAUSAL_LM"))
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR)
    sync(model)
    tasks = [(d, t) for d in DOMAINS for t in get_tasks(d, task_split_name="train")]

    for epoch in range(EPOCHS):
        random.shuffle(tasks)
        for step in range(0, len(tasks), TASKS_PER_STEP):
            jobs = [(d, t) for d, t in tasks[step:step + TASKS_PER_STEP] for _ in range(G)]
            with ThreadPoolExecutor(len(jobs)) as pool:
                runs = list(pool.map(lambda j: rollout(*j, enforce=True, model="policy", temperature=1.0,
                                                       seed=random.randrange(2**31)), jobs))

            # group-relative advantages, R = r_task - β·B
            R = torch.tensor([r - BETA * agent.violations for r, agent in runs]).view(-1, G)
            A = ((R - R.mean(1, keepdim=True)) / (R.std(1, keepdim=True) + 1e-4)).flatten()
            batch = [(p, c, a) for (_, agent), a in zip(runs, A.tolist()) if a != 0 for p, c in agent.samples]
            if not batch:
                continue

            model.eval()
            with torch.no_grad():
                old = [token_logprobs(model, p, c) for p, c, _ in batch]
                with model.disable_adapter():
                    ref = [token_logprobs(model, p, c) for p, c, _ in batch]

            model.train()
            for _ in range(MU):
                order = random.sample(range(len(batch)), len(batch))
                for i in range(0, len(order), MINIBATCH):
                    mb = order[i:i + MINIBATCH]
                    ntok = sum(len(batch[j][1]) for j in mb)
                    opt.zero_grad()
                    for j in mb:
                        p, c, a = batch[j]
                        logp = token_logprobs(model, p, c)
                        ratio = (logp - old[j]).exp()
                        surr = torch.min(ratio * a, ratio.clamp(1 - CLIP, 1 + CLIP) * a)
                        kl = (ref[j] - logp).exp() - (ref[j] - logp) - 1
                        (-(surr - KL_COEF * kl).sum() / ntok).backward()
                    opt.step()

            sync(model)
            print(f"epoch {epoch} step {step // TASKS_PER_STEP} R={R.mean(1).tolist()}", flush=True)
        model.save_pretrained(f"ckpt/epoch{epoch}")


if __name__ == "__main__":
    main()
