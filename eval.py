import argparse
import json
from concurrent.futures import ThreadPoolExecutor

from tau2.run import get_tasks

from agent import DOMAINS, MODEL, rollout


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=MODEL, help="served model name; 'trained' for the LoRA policy")
    p.add_argument("--trials", type=int, default=1)
    args = p.parse_args()

    results = {}
    for domain in DOMAINS:
        jobs = [(task, trial) for task in get_tasks(domain, task_split_name="test") for trial in range(args.trials)]
        with ThreadPoolExecutor(16) as pool:
            runs = list(pool.map(lambda j: rollout(domain, j[0], enforce=False, model=args.model, temperature=0.0, seed=j[1]), jobs))
        results[domain] = {
            "task_success": sum(r == 1.0 for r, _ in runs) / len(runs),
            "conformance": sum(agent.violations == 0 for _, agent in runs) / len(runs),
        }
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
