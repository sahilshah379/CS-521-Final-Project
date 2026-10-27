import json
import re
import threading
import uuid

import requests
from transformers import AutoTokenizer

from tau2.agent.llm_agent import LLMAgent
from tau2.data_model.message import AssistantMessage, MultiToolMessage, ToolCall, ToolMessage, UserMessage
from tau2.data_model.simulation import TextRunConfig
from tau2.run import build_text_orchestrator, run_simulation

from rules import Checker

MODEL = "Qwen/Qwen2.5-7B-Instruct"
VLLM_URL = "http://localhost:8000"
USER_LLM = "gpt-4.1-mini"
DOMAINS = ["airline", "retail"]
MAX_STEPS, MAX_RETRIES, MAX_NEW_TOKENS = 100, 3, 1024
TOOL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.S)
TOK = AutoTokenizer.from_pretrained(MODEL)
TOK_LOCK = threading.Lock()


def to_hf(messages):
    out = []
    for m in messages:
        d = {"role": m.role, "content": m.content or ""}
        if getattr(m, "tool_calls", None):
            d["tool_calls"] = [{"type": "function", "function": {"name": c.name, "arguments": c.arguments}} for c in m.tool_calls]
        out.append(d)
    return out


def parse(text):
    try:
        calls = [ToolCall(id=f"call_{uuid.uuid4().hex[:8]}", name=c["name"], arguments=c.get("arguments", {}))
                 for c in map(json.loads, TOOL_RE.findall(text))]
    except (json.JSONDecodeError, KeyError, TypeError):
        calls = []
    return AssistantMessage(role="assistant", content=None if calls else text.strip(), tool_calls=calls or None)


class ShieldedAgent(LLMAgent):
    """Policy served by vLLM whose proposed tool calls pass through the TL checker."""

    def __init__(self, tools, domain_policy, checker, enforce, model, temperature):
        super().__init__(tools=tools, domain_policy=domain_policy, llm=model)
        self.checker, self.enforce, self.temperature = checker, enforce, temperature
        self.schemas = [t.openai_schema for t in tools]
        self.samples, self.violations = [], 0

    def _generate(self, messages):
        with TOK_LOCK:
            text = TOK.apply_chat_template(to_hf(messages), tools=self.schemas, add_generation_prompt=True, tokenize=False)
            prompt = TOK(text, add_special_tokens=False).input_ids
        r = requests.post(f"{VLLM_URL}/v1/completions", json={
            "model": self.llm, "prompt": prompt, "max_tokens": MAX_NEW_TOKENS,
            "temperature": self.temperature, "return_token_ids": True}).json()["choices"][0]
        self.samples.append((prompt, r["token_ids"]))
        return parse(r["text"])

    def generate_next_message(self, message, state):
        if isinstance(message, MultiToolMessage):
            state.messages.extend(message.tool_messages)
        else:
            state.messages.append(message)
            if isinstance(message, UserMessage):
                self.checker.observe_user(message.content)
        for _ in range(MAX_RETRIES):
            reply = self._generate(state.system_messages + state.messages)
            violated = self.checker.check(reply.tool_calls or [])
            if not violated:
                break
            self.violations += 1
            if not self.enforce:
                break
            state.messages += [reply, *(ToolMessage(id=c.id, role="tool", content=f"Blocked: {violated}.", error=True)
                                        for c in reply.tool_calls)]
        else:
            reply = AssistantMessage(role="assistant", content="I'm sorry, I can't complete that action right now.")
        self.checker.record(reply.tool_calls or [])
        state.messages.append(reply)
        return reply, state


def rollout(domain, task, enforce, model, temperature, seed):
    config = TextRunConfig(domain=domain, llm_user=USER_LLM, llm_args_user={"temperature": 0.0}, max_steps=MAX_STEPS)
    orch = build_text_orchestrator(config, task, seed=seed)
    orch.agent = agent = ShieldedAgent(orch.agent.tools, orch.agent.domain_policy, Checker(domain), enforce, model, temperature)
    sim = run_simulation(orch)
    return (sim.reward_info.reward if sim.reward_info else 0.0), agent
