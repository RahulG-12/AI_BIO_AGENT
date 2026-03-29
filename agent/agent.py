"""
agent/agent.py

The Biological Age Agent — orchestrates tool calls to produce
a comprehensive aging assessment for a given subject.

Architecture:
  User prompt → GPT-4o with function calling → tool execution loop
  → hallucination check via RAG grounding → final synthesis
"""

import os
import json
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

from agent.tools import TOOL_DEFINITIONS, TOOL_FUNCTIONS
from rag.retriever import LongevityRetriever

SYSTEM_PROMPT = """You are a Biological Age Analysis Agent built for Immortigen, a longevity biotech
company building the Digital Twin of the Human Body.

Your role: analyse a subject's biological age data and produce a comprehensive,
evidence-based longevity assessment.

You have access to 5 tools:
1. predict_biological_age — compute composite biological age
2. search_longevity_papers — find relevant research (ALWAYS cite papers in your response)
3. suggest_interventions — get personalised intervention recommendations
4. explain_biomarker — explain what a specific marker means for aging
5. flag_abnormal_biomarkers — identify out-of-range values

Workflow for a full assessment:
1. Flag abnormal biomarkers first
2. Predict composite biological age
3. Search for relevant research supporting your findings
4. Suggest personalised interventions
5. Synthesise everything into a structured report

Important:
- Always cite the research papers returned by search_longevity_papers
- Be precise about biological mechanisms — avoid vague wellness language
- Clearly distinguish between accelerated vs. decelerated aging
- Note limitations and confidence levels honestly
- Your audience is scientifically literate
"""


class BiologicalAgeAgent:
    """
    Agentic loop that processes a subject's data through multiple tool calls
    and produces a comprehensive aging report.
    """

    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = os.getenv("OPENAI_MODEL", "gpt-4o")
        self.retriever = LongevityRetriever()
        self.retriever.build()
        self.conversation_history = []

    def reset(self):
        self.conversation_history = []

    def _run_tool(self, tool_call) -> str:
        """Execute a tool call and return JSON string result."""
        fn_name = tool_call.function.name
        try:
            args = json.loads(tool_call.function.arguments)
        except json.JSONDecodeError:
            return json.dumps({"error": "Invalid JSON arguments"})

        if fn_name not in TOOL_FUNCTIONS:
            return json.dumps({"error": f"Unknown tool: {fn_name}"})

        try:
            result = TOOL_FUNCTIONS[fn_name](**args)
            return json.dumps(result)
        except Exception as e:
            return json.dumps({"error": str(e)})

    def analyse(self, subject_data: dict, user_question: str | None = None) -> str:
        """
        Run a full biological age analysis for a subject.

        subject_data: {
            chronological_age: float,
            methylation_age: float (optional),
            blood_age: float (optional),
            biomarkers: dict (optional),
            sample_id: str (optional),
        }
        """
        self.reset()

        # Build the user message
        data_str = json.dumps(subject_data, indent=2)
        prompt = (
            f"Please perform a complete biological age assessment for this subject.\n\n"
            f"Subject Data:\n{data_str}\n\n"
        )
        if user_question:
            prompt += f"Specific question: {user_question}\n\n"
        prompt += "Use the available tools to produce a comprehensive, evidence-based report."

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]

        # Agentic tool-call loop
        max_iterations = 10
        for iteration in range(max_iterations):
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=TOOL_DEFINITIONS,
                tool_choice="auto",
                temperature=0.2,
            )

            message = response.choices[0].message
            messages.append(message)

            if response.choices[0].finish_reason == "tool_calls":
                # Execute all tool calls in parallel (sequential here for simplicity)
                for tool_call in message.tool_calls:
                    result = self._run_tool(tool_call)
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": result,
                    })
            else:
                # Model finished — return final response
                final = message.content

                # Hallucination check: verify key claims are grounded
                grounding = self._check_grounding(final)
                if grounding["ungrounded_count"] > 0:
                    final += (
                        f"\n\n---\n*Grounding note: {grounding['ungrounded_count']} claim(s) "
                        f"could not be fully verified against the research corpus. "
                        f"Please verify these independently.*"
                    )

                self.conversation_history = messages
                return final

        return "Analysis incomplete — maximum iterations reached."

    def chat(self, user_message: str) -> str:
        """
        Continue a conversation — ask follow-up questions about a previous analysis.
        """
        if not self.conversation_history:
            return self.analyse({}, user_message)

        self.conversation_history.append({"role": "user", "content": user_message})

        response = self.client.chat.completions.create(
            model=self.model,
            messages=self.conversation_history,
            tools=TOOL_DEFINITIONS,
            tool_choice="auto",
            temperature=0.2,
        )

        message = response.choices[0].message
        self.conversation_history.append(message)

        if response.choices[0].finish_reason == "tool_calls":
            for tool_call in message.tool_calls:
                result = self._run_tool(tool_call)
                self.conversation_history.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result,
                })
            # Get final response after tool execution
            follow_up = self.client.chat.completions.create(
                model=self.model,
                messages=self.conversation_history,
                temperature=0.2,
            )
            reply = follow_up.choices[0].message.content
            self.conversation_history.append(follow_up.choices[0].message)
            return reply
        else:
            return message.content

    def _check_grounding(self, text: str, sample_claims: int = 3) -> dict:
        """
        Sample key claims from the response and verify they are grounded in the corpus.
        Simple heuristic: check sentences containing quantitative claims.
        """
        sentences = [s.strip() for s in text.split(".") if len(s.strip()) > 50]
        # Sample sentences that look like claims (contain numbers or key terms)
        claim_sentences = [
            s for s in sentences
            if any(char.isdigit() for char in s) or
               any(kw in s.lower() for kw in ["associated", "evidence", "shown", "study", "research"])
        ][:sample_claims]

        ungrounded = []
        for claim in claim_sentences:
            check = self.retriever.verify_claim(claim)
            if not check["grounded"]:
                ungrounded.append(claim[:100])

        return {
            "checked_claims": len(claim_sentences),
            "ungrounded_count": len(ungrounded),
            "ungrounded_samples": ungrounded,
        }
