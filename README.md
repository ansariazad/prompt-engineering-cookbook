# 🍳 Prompt Engineering Cookbook — 10 LLM Prompting Patterns in Pure Python

> **Every major prompting technique — Zero-Shot to Tree of Thoughts — implemented, runnable, and documented.**

A single-file reference implementation of the 10 most important LLM prompting patterns. No API keys required — uses Ollama locally or runs deterministic fallbacks offline. Covers Zero-Shot, Few-Shot, Chain-of-Thought, Self-Consistency, Tree of Thoughts, ReAct, Meta-Prompting, Role Prompting, Prompt Chaining, and Auto Prompt Optimization.

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)](https://python.org)
[![Zero Deps](https://img.shields.io/badge/Core-Zero_Dependencies-22c55e)](https://github.com/ansariazad/prompt-engineering-cookbook)
[![Ollama](https://img.shields.io/badge/Ollama-llama3.2-F97316)](https://ollama.ai)
[![License: MIT](https://img.shields.io/badge/License-MIT-22c55e)](LICENSE)
[![Stars](https://img.shields.io/github/stars/ansariazad/prompt-engineering-cookbook?style=social)](https://github.com/ansariazad/prompt-engineering-cookbook/stargazers)

---

## 📺 Demo

```
$ python3 06_prompt_engineering_cookbook.py

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  PROMPT ENGINEERING COOKBOOK — Engine: offline-fallback
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

[1] ZERO-SHOT          → Direct answer to: Explain attention mechanism...
[2] FEW-SHOT (k=3)     → Answer for: What is layer normalization?
[3] CHAIN-OF-THOUGHT   → Step 1: 7B × 2 bytes = 14 GB GPU memory
[4] SELF-CONSISTENCY   → Majority answer (5 paths): 36.0
[5] TREE OF THOUGHTS   → Best path score=0.714, depth=2
[6] REACT              → Thought → Action → Observation × 3 steps
[7] META-PROMPTING     → LLM generated its own prompt, then answered
[8] ROLE PROMPTING     → ml_engineer: "The tradeoff is compute vs memory..."
[9] PROMPT CHAINING    → outline → draft → critique → final
[10] AUTO OPTIMIZER    → Score improved 0.40 → 0.80 in 3 iterations
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

---

## 📑 Table of Contents

- [The 10 Patterns](#-the-10-patterns)
- [Quick Start](#-quick-start)
- [Pattern Deep Dives](#-pattern-deep-dives)
- [When to Use Which Pattern](#-when-to-use-which-pattern)
- [Roadmap](#-roadmap)
- [References](#-references)

---

## 🛠️ The 10 Patterns

| # | Pattern | Key Idea | Best For |
|---|---------|----------|----------|
| 1 | **Zero-Shot** | Direct instruction, no examples | Simple factual tasks |
| 2 | **Few-Shot** | k demonstrations before the query | Classification, format matching |
| 3 | **Chain-of-Thought** | "Let's think step by step" | Math, logic, multi-step reasoning |
| 4 | **Self-Consistency CoT** | Majority vote over N paths | High-stakes reasoning |
| 5 | **Tree of Thoughts** | BFS/DFS over reasoning branches | Planning, search problems |
| 6 | **ReAct** | Reason + Act + Observe loop | Tool-use, agent tasks |
| 7 | **Meta-Prompting** | LLM writes its own prompt | Unknown optimal prompt |
| 8 | **Role Prompting** | Assign expert persona | Domain-specific answers |
| 9 | **Prompt Chaining** | Sequential refinement pipeline | Long-form content |
| 10 | **Auto Optimizer** | Score + rewrite loop | Production prompt tuning |

---

## 🚀 Quick Start

```bash
# Clone
git clone https://github.com/ansariazad/prompt-engineering-cookbook.git
cd prompt-engineering-cookbook

# Zero dependencies — run offline
python3 06_prompt_engineering_cookbook.py

# With real Ollama LLM
brew install ollama && ollama pull llama3.2
OLLAMA_MODEL=llama3.2 python3 06_prompt_engineering_cookbook.py
```

---

## 🔬 Pattern Deep Dives

### Chain-of-Thought
```python
from cookbook import chain_of_thought

reasoning, answer = chain_of_thought(
    "If a 7B model uses float16, how much GPU memory does it need?"
)
# reasoning: "Step 1: 7B parameters × 2 bytes = 14 GB..."
# answer: "14 GB minimum"
```

### Tree of Thoughts
```python
from cookbook import tree_of_thoughts

result = tree_of_thoughts(
    problem="Design a RAG system for medical Q&A",
    breadth=3,   # candidates per level
    depth=2      # levels to explore
)
# BFS expands best-scoring thought branches
```

### ReAct Loop
```python
from cookbook import react_loop

tools = {
    "search": lambda q: f"Results for {q}: ...",
    "calculator": lambda q: f"= {eval(q)}",
}
result = react_loop("What is 15% of the GDP of India?", tools, max_steps=4)
# Thought → search[India GDP] → Observe → Thought → calculator[...] → Answer
```

### Auto Prompt Optimizer
```python
from cookbook import auto_prompt_optimizer

result = auto_prompt_optimizer(
    task="Explain RLHF to a junior engineer",
    rubric_keywords=["reward", "human", "feedback", "PPO", "policy"],
    max_iterations=3,
    target_score=0.8,
)
print(result["best_score"])   # 0.80
print(result["iterations"])   # 2
```

---

## 📊 When to Use Which Pattern

```
Task type?
├── Simple factual    → Zero-Shot
├── Format matching   → Few-Shot (k=3-5)
├── Multi-step logic  → Chain-of-Thought
│   └── High-stakes   → Self-Consistency CoT (n=5-10)
├── Planning/search   → Tree of Thoughts
├── Tool-use agents   → ReAct
├── Unknown prompt    → Meta-Prompting
├── Domain expertise  → Role Prompting
├── Long-form content → Prompt Chaining
└── Production tuning → Auto Optimizer
```

---

## 🗺️ Roadmap

- [ ] Automatic Few-Shot example selection (via embedding similarity)
- [ ] Constitutional AI self-critique pattern
- [ ] Structured output prompting (JSON schema enforcement)
- [ ] Multi-agent debate pattern
- [ ] DSPy-style compiled prompt optimization

---

## 📚 References

| Paper | Authors |
|-------|---------|
| [Chain-of-Thought Prompting Elicits Reasoning](https://arxiv.org/abs/2201.11903) | Wei et al., 2022 |
| [Self-Consistency Improves CoT Reasoning](https://arxiv.org/abs/2203.11171) | Wang et al., 2022 |
| [Tree of Thoughts](https://arxiv.org/abs/2305.10601) | Yao et al., 2023 |
| [ReAct: Synergizing Reasoning and Acting](https://arxiv.org/abs/2210.03629) | Yao et al., 2022 |
| [Large Language Models are Zero-Shot Reasoners](https://arxiv.org/abs/2205.11916) | Kojima et al., 2022 |

---

## ⭐ Support

Found this useful? **Star the repo** to help others discover these patterns!

[![GitHub stars](https://img.shields.io/github/stars/ansariazad/prompt-engineering-cookbook?style=for-the-badge)](https://github.com/ansariazad/prompt-engineering-cookbook/stargazers)

---

## 📄 License

MIT © 2026

---

## 👤 Author

**Azad Ansari** — Python Developer & AI Engineer, Mumbai

[![GitHub](https://img.shields.io/badge/GitHub-ansariazad-181717?logo=github)](https://github.com/ansariazad)
[![Portfolio](https://img.shields.io/badge/Portfolio-ansariazad.github.io-blue)](https://ansariazad.github.io)
