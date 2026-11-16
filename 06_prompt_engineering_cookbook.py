#!/usr/bin/env python3
"""
06_prompt_engineering_cookbook.py — Advanced LLM Prompting Patterns from Scratch
==================================================================================
Reference Sources:
  - Wei et al., "Chain-of-Thought Prompting Elicits Reasoning in LLMs" (2022)
    https://arxiv.org/abs/2201.11903
  - Yao et al., "Tree of Thoughts: Deliberate Problem Solving with LLMs" (2023)
    https://arxiv.org/abs/2305.10601
  - Yao et al., "ReAct: Synergizing Reasoning and Acting in LLMs" (2022)
    https://arxiv.org/abs/2210.03629

Implemented Patterns:
  1. Zero-Shot Prompting        — direct instruction, no examples
  2. Few-Shot Prompting         — in-context learning with k examples
  3. Chain-of-Thought (CoT)     — "Let's think step by step"
  4. Self-Consistency CoT       — majority vote across N reasoning paths
  5. Tree of Thoughts (ToT)     — BFS/DFS over reasoning branches
  6. ReAct                      — interleaved Reason + Act + Observe loop
  7. Meta-Prompting              — LLM generates its own prompt
  8. Role Prompting              — persona assignment for domain expertise
  9. Prompt Chaining             — sequential refinement pipeline
 10. Automatic Prompt Optimizer — score-and-rewrite loop

Usage:
  python3 06_prompt_engineering_cookbook.py
  OLLAMA_MODEL=llama3.2 python3 06_prompt_engineering_cookbook.py
"""

from __future__ import annotations

import json
import os
import random
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# LLM Backend (Ollama or deterministic offline fallback)
# ---------------------------------------------------------------------------

def call_llm(prompt: str, fallback: str, system: str = "") -> str:
    """
    Calls local Ollama when OLLAMA_MODEL env var is set.
    Falls back to deterministic offline string — always works without any API key.
    """
    model = os.environ.get("OLLAMA_MODEL", "")
    if model:
        try:
            payload = {"model": model, "prompt": prompt, "stream": False}
            if system:
                payload["system"] = system
            req = urllib.request.Request(
                "http://localhost:11434/api/generate",
                data=json.dumps(payload).encode(),
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=15) as resp:
                return str(json.loads(resp.read().decode()).get("response", "")).strip()
        except Exception:
            pass
    return fallback


# ---------------------------------------------------------------------------
# 1. Zero-Shot Prompting
# ---------------------------------------------------------------------------

def zero_shot(task: str) -> str:
    """Direct instruction with no examples."""
    prompt = f"Task: {task}\n\nAnswer:"
    return call_llm(prompt, fallback=f"[Zero-Shot] Direct answer to: {task[:50]}")


# ---------------------------------------------------------------------------
# 2. Few-Shot Prompting (In-Context Learning)
# ---------------------------------------------------------------------------

@dataclass
class Example:
    input: str
    output: str


def few_shot(task: str, examples: List[Example], k: int = 3) -> str:
    """In-context learning: prepend k demonstrations before the actual query."""
    shots = examples[:k]
    demo_block = "\n\n".join(
        f"Input: {ex.input}\nOutput: {ex.output}" for ex in shots
    )
    prompt = f"{demo_block}\n\nInput: {task}\nOutput:"
    return call_llm(
        prompt,
        fallback=f"[Few-Shot k={k}] Answer for: {task[:50]} (learned from {k} examples)",
    )


# ---------------------------------------------------------------------------
# 3. Chain-of-Thought (CoT)
# ---------------------------------------------------------------------------

def chain_of_thought(question: str) -> Tuple[str, str]:
    """
    Appends "Let's think step by step." to elicit explicit reasoning.
    Returns (reasoning_chain, final_answer).
    """
    prompt = f"Q: {question}\nA: Let's think step by step.\n"
    response = call_llm(
        prompt,
        fallback=(
            "Step 1: Identify the key facts.\n"
            "Step 2: Apply relevant reasoning.\n"
            "Step 3: Arrive at the conclusion.\n"
            "Therefore, the answer is: 42."
        ),
    )
    lines = response.strip().split("\n")
    reasoning = "\n".join(lines[:-1]) if len(lines) > 1 else response
    answer = lines[-1] if len(lines) > 1 else response
    return reasoning, answer


# ---------------------------------------------------------------------------
# 4. Self-Consistency CoT
# ---------------------------------------------------------------------------

def self_consistency_cot(question: str, n_paths: int = 5) -> str:
    """
    Samples N independent CoT reasoning paths, then takes majority vote on the answer.
    Significantly more reliable than single-path CoT for math/logic problems.
    """
    answers: List[str] = []
    for i in range(n_paths):
        _, answer = chain_of_thought(question)
        # In offline mode, simulate slight variation
        answers.append(answer.strip() or f"answer_{i % 3}")

    # Majority vote
    from collections import Counter
    winner = Counter(answers).most_common(1)[0][0]
    return f"[Self-Consistency n={n_paths}] Majority answer: {winner}"


# ---------------------------------------------------------------------------
# 5. Tree of Thoughts (ToT)
# ---------------------------------------------------------------------------

@dataclass
class ThoughtNode:
    thought: str
    score: float
    children: List["ThoughtNode"] = field(default_factory=list)
    depth: int = 0


def _score_thought(thought: str, goal: str) -> float:
    """Heuristic: score how relevant a thought is to the goal."""
    keywords = set(goal.lower().split())
    thought_words = set(thought.lower().split())
    overlap = len(keywords & thought_words)
    return overlap / max(len(keywords), 1)


def tree_of_thoughts(problem: str, breadth: int = 3, depth: int = 2) -> str:
    """
    BFS over a tree of reasoning steps. At each level, generates `breadth` candidate
    thoughts, scores them, and expands the best ones up to `depth` levels.
    """
    candidate_thoughts = [
        f"Consider the problem from angle {i+1}: {problem[:40]}..." for i in range(breadth)
    ]

    # Build root nodes
    roots = [
        ThoughtNode(thought=t, score=_score_thought(t, problem), depth=0)
        for t in candidate_thoughts
    ]

    best_path: List[str] = []
    best_score = -1.0

    # BFS expansion
    frontier = sorted(roots, key=lambda n: n.score, reverse=True)[:breadth]
    for d in range(depth):
        next_frontier = []
        for node in frontier:
            expansions = [
                ThoughtNode(
                    thought=f"[Depth {d+1}] Expanding: {node.thought[:40]} → sub-thought {j+1}",
                    score=_score_thought(node.thought, problem) * (0.9 ** (d + 1)),
                    depth=d + 1,
                )
                for j in range(breadth)
            ]
            node.children = expansions
            next_frontier.extend(expansions)

        frontier = sorted(next_frontier, key=lambda n: n.score, reverse=True)[:breadth]

    # Return highest-scoring leaf
    if frontier:
        best_node = frontier[0]
        return f"[ToT breadth={breadth} depth={depth}] Best path score={best_node.score:.3f}\n{best_node.thought}"
    return "[ToT] No solution found"


# ---------------------------------------------------------------------------
# 6. ReAct (Reason + Act + Observe)
# ---------------------------------------------------------------------------

def react_loop(question: str, tools: Dict[str, Callable[[str], str]], max_steps: int = 4) -> str:
    """
    Interleaves Thought → Action → Observation steps.
    Stops when the model produces a 'Final Answer:' or max_steps is reached.
    """
    history: List[str] = [f"Question: {question}"]
    tool_names = list(tools.keys())

    for step in range(max_steps):
        # Thought
        thought = call_llm(
            "\n".join(history) + f"\nThought {step+1}:",
            fallback=f"I need to look up information using {tool_names[step % len(tool_names)]}.",
        )
        history.append(f"Thought {step+1}: {thought}")

        # Action (pick a tool)
        tool_name = tool_names[step % len(tool_names)]
        query = question.split()[:3]
        action_input = " ".join(query)
        history.append(f"Action {step+1}: {tool_name}[{action_input}]")

        # Observation
        observation = tools[tool_name](action_input)
        history.append(f"Observation {step+1}: {observation}")

        if "Final Answer:" in thought or step == max_steps - 1:
            break

    final = call_llm(
        "\n".join(history) + "\nFinal Answer:",
        fallback=f"Based on observations, the answer is: [synthesized from {step+1} steps]",
    )
    history.append(f"Final Answer: {final}")
    return "\n".join(history)


# ---------------------------------------------------------------------------
# 7. Meta-Prompting (LLM generates its own prompt)
# ---------------------------------------------------------------------------

def meta_prompt(task_description: str) -> Tuple[str, str]:
    """
    Step 1: Ask the LLM to generate the best prompt for a given task.
    Step 2: Execute that generated prompt.
    Returns (generated_prompt, final_response).
    """
    meta = (
        f"You are a prompt engineering expert. Write the single best prompt to accomplish:\n"
        f"Task: {task_description}\n\n"
        f"Output only the prompt, nothing else."
    )
    generated_prompt = call_llm(
        meta,
        fallback=(
            f"You are an expert. Carefully analyze the following and provide a detailed, "
            f"accurate response: {task_description}"
        ),
    )
    response = call_llm(generated_prompt, fallback=f"[Meta] Response to self-generated prompt for: {task_description[:40]}")
    return generated_prompt, response


# ---------------------------------------------------------------------------
# 8. Role Prompting
# ---------------------------------------------------------------------------

PERSONAS = {
    "ml_engineer": (
        "You are a senior machine learning engineer at Google DeepMind with 10 years of "
        "experience in transformers, LLMs, and distributed training. You give precise, "
        "technical answers with code examples."
    ),
    "product_manager": (
        "You are a sharp product manager who thinks in terms of user impact, metrics, "
        "and business value. Keep answers concise and focused on outcomes."
    ),
    "researcher": (
        "You are an AI safety researcher. You reason carefully, cite papers, and always "
        "consider potential failure modes and second-order effects."
    ),
}


def role_prompt(question: str, persona: str = "ml_engineer") -> str:
    """Assigns a detailed persona to the LLM before asking the question."""
    system = PERSONAS.get(persona, PERSONAS["ml_engineer"])
    prompt = f"Question: {question}\n\nAnswer:"
    return call_llm(prompt, system=system, fallback=f"[Role: {persona}] Expert answer to: {question[:50]}")


# ---------------------------------------------------------------------------
# 9. Prompt Chaining (Sequential Refinement)
# ---------------------------------------------------------------------------

def prompt_chain(task: str) -> Dict[str, str]:
    """
    Chains 4 sequential prompts: outline → draft → critique → final.
    Each step's output feeds the next step's input.
    """
    # Step 1: Outline
    outline = call_llm(
        f"Create a structured outline for: {task}",
        fallback="1. Introduction\n2. Core concepts\n3. Implementation\n4. Examples\n5. Conclusion",
    )

    # Step 2: Draft from outline
    draft = call_llm(
        f"Using this outline:\n{outline}\n\nWrite a detailed draft for: {task}",
        fallback=f"[Draft] Expanded content based on outline for: {task[:40]}",
    )

    # Step 3: Critique the draft
    critique = call_llm(
        f"Critique this draft and identify improvements:\n{draft}",
        fallback="The draft is clear but could use more concrete examples and tighter structure.",
    )

    # Step 4: Final version incorporating critique
    final = call_llm(
        f"Rewrite the draft incorporating this critique:\nCritique: {critique}\nDraft: {draft}",
        fallback=f"[Final] Polished version of: {task[:40]}",
    )

    return {"outline": outline, "draft": draft, "critique": critique, "final": final}


# ---------------------------------------------------------------------------
# 10. Automatic Prompt Optimizer
# ---------------------------------------------------------------------------

def _score_response(response: str, rubric_keywords: List[str]) -> float:
    """Simple keyword-based quality scorer (0.0 – 1.0)."""
    response_lower = response.lower()
    hits = sum(1 for kw in rubric_keywords if kw.lower() in response_lower)
    return hits / max(len(rubric_keywords), 1)


def auto_prompt_optimizer(
    task: str,
    rubric_keywords: List[str],
    max_iterations: int = 3,
    target_score: float = 0.75,
) -> Dict[str, Any]:
    """
    Iteratively improves a prompt by scoring the response and rewriting the prompt
    if the score is below target. Returns the best prompt, response, and score found.
    """
    prompt = f"Answer the following: {task}"
    best_prompt = prompt
    best_response = ""
    best_score = 0.0
    history = []

    for iteration in range(max_iterations):
        response = call_llm(prompt, fallback=f"Response to: {task[:40]}")
        score = _score_response(response, rubric_keywords)
        history.append({"iteration": iteration + 1, "prompt": prompt, "score": score})

        if score > best_score:
            best_score = score
            best_prompt = prompt
            best_response = response

        if score >= target_score:
            break

        # Rewrite the prompt to target missing keywords
        missing = [kw for kw in rubric_keywords if kw.lower() not in response.lower()]
        if missing:
            prompt = (
                f"Answer the following, making sure to address: {', '.join(missing[:3])}.\n"
                f"Task: {task}"
            )

    return {
        "best_prompt": best_prompt,
        "best_response": best_response,
        "best_score": best_score,
        "iterations": len(history),
        "history": history,
    }


# ---------------------------------------------------------------------------
# Demo tools for ReAct
# ---------------------------------------------------------------------------

DEMO_TOOLS = {
    "search": lambda q: f"Search results for '{q}': [AI paper on transformers, 2024]",
    "calculator": lambda q: f"calc({q}) = {sum(ord(c) for c in q) % 100}",
    "lookup": lambda q: f"Knowledge base: '{q}' refers to a core concept in deep learning",
}


# ---------------------------------------------------------------------------
# Main — demo all 10 patterns
# ---------------------------------------------------------------------------

def main() -> None:
    sep = "━" * 60
    engine = os.environ.get("OLLAMA_MODEL", "offline-fallback")
    print(f"\n{'━'*60}")
    print(f"  PROMPT ENGINEERING COOKBOOK — Engine: {engine}")
    print(f"{'━'*60}\n")

    # 1. Zero-Shot
    print(f"[1] ZERO-SHOT PROMPTING\n{sep}")
    result = zero_shot("Explain what attention mechanism does in a transformer")
    print(result[:200])

    # 2. Few-Shot
    print(f"\n[2] FEW-SHOT PROMPTING (k=3)\n{sep}")
    examples = [
        Example("What is a neuron?", "A neuron is the basic computational unit of a neural network."),
        Example("What is backprop?", "Backpropagation computes gradients via the chain rule."),
        Example("What is dropout?", "Dropout randomly zeroes activations during training to reduce overfitting."),
    ]
    result = few_shot("What is layer normalization?", examples, k=3)
    print(result[:200])

    # 3. Chain-of-Thought
    print(f"\n[3] CHAIN-OF-THOUGHT\n{sep}")
    reasoning, answer = chain_of_thought("If a model has 7B parameters and each is float16, how much GPU memory does it need?")
    print(f"Reasoning:\n{reasoning[:200]}")
    print(f"Answer: {answer[:100]}")

    # 4. Self-Consistency
    print(f"\n[4] SELF-CONSISTENCY CoT (n=5)\n{sep}")
    result = self_consistency_cot("What is 15% of 240?", n_paths=5)
    print(result)

    # 5. Tree of Thoughts
    print(f"\n[5] TREE OF THOUGHTS (breadth=3, depth=2)\n{sep}")
    result = tree_of_thoughts("How should I structure a RAG system for a medical Q&A bot?", breadth=3, depth=2)
    print(result[:300])

    # 6. ReAct
    print(f"\n[6] REACT (Reason + Act + Observe)\n{sep}")
    result = react_loop("What are the latest advances in transformer efficiency?", DEMO_TOOLS, max_steps=3)
    print(result[:400])

    # 7. Meta-Prompting
    print(f"\n[7] META-PROMPTING\n{sep}")
    gen_prompt, response = meta_prompt("Explain LoRA fine-tuning to a junior ML engineer")
    print(f"Generated prompt: {gen_prompt[:150]}")
    print(f"Response: {response[:150]}")

    # 8. Role Prompting
    print(f"\n[8] ROLE PROMPTING (persona=ml_engineer)\n{sep}")
    result = role_prompt("What are the tradeoffs between full fine-tuning and LoRA?", persona="ml_engineer")
    print(result[:250])

    # 9. Prompt Chaining
    print(f"\n[9] PROMPT CHAINING (4-step pipeline)\n{sep}")
    chain = prompt_chain("Write a guide to building production LLM applications")
    for step, content in chain.items():
        print(f"  [{step.upper()}] {content[:80]}...")

    # 10. Auto Prompt Optimizer
    print(f"\n[10] AUTO PROMPT OPTIMIZER\n{sep}")
    result = auto_prompt_optimizer(
        task="Explain how RLHF trains ChatGPT",
        rubric_keywords=["reward", "human", "feedback", "policy", "PPO"],
        max_iterations=3,
        target_score=0.6,
    )
    print(f"Best score : {result['best_score']:.2f}")
    print(f"Iterations : {result['iterations']}")
    print(f"Best prompt: {result['best_prompt'][:120]}")

    print(f"\n{'━'*60}")
    print("  All 10 prompting patterns demonstrated.")
    print(f"{'━'*60}\n")


if __name__ == "__main__":
    main()
