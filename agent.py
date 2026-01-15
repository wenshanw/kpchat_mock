import os

import mlflow
import openai

from openai import OpenAI

import pyspark.sql.functions as F

from datetime import date, timedelta
import random

from typing import Any, Dict, List, cast, Generator

from databricks.sdk import WorkspaceClient
from mlflow.entities import SpanType
from mlflow.pyfunc import ResponsesAgent

from mlflow.types.responses import (
    ResponsesAgentRequest,
    ResponsesAgentResponse,
    ResponsesAgentStreamEvent,
    output_to_responses_items_stream,
    to_chat_completions_input,
)
from openai import OpenAI
from pydantic import BaseModel


# Load system prompt
# SYSTEM_PROMPT = mlflow.genai.load_prompt("prompts:/genai_apps.kpchat_mock.kpchat_system/7")
SYSTEM_PROMPT = """
You are an assistant agent with a specific focus on providing information related to drug pricing and delivery dates. Your role is strictly limited to these domains, which means you should avoid providing information or assistance outside of these areas. Below are your guidelines:\n\n1. **Domain-Specific Knowledge**: You are equipped to answer inquiries specifically about the pricing of drugs and the expected delivery dates for pharmaceuticals.\n\n2. **Knowledge Limitation**: If a user inquiry extends beyond drug pricing and delivery dates\u2014such as inquiries regarding medical procedures costs, copayments for healthcare services, or contact information\u2014you should respond by expressing your limitation in scope. For example, use a standard response such as: \"Sorry, I don't know. I can only answer questions related to drug pricing or delivery dates.\"\n\n3. **Consistent Response Protocol**: Maintain a polite and concise tone, acknowledging the user\u2019s inquiry while redirecting them to seek further assistance elsewhere if needed.\n\n4. **Clarification Queries**: If a user\u2019s query is vague or you need more details specifically related to drug pricing or delivery dates, feel free to request additional information. Otherwise, reinforce your limited scope.\n\nBy adhering to these guidelines, you ensure that the assistance provided is accurate and within the boundaries of your current role.
"""

class SimpleResponsesAgent(ResponsesAgent):
    def __init__(self, model: str):
        self.client = OpenAI()
        self.model = model


    def get_drug_price(self, drug_name: str) -> str:
        """Mock tool to get drug pricing."""
        return f"${15.99}"
    

    def check_delivery_date(self, drug_name: str, user_name: str) -> str:
        """Mock tool to check delivery date."""
        max_days_ahead = 100
        start = date.today() + timedelta(days=1)
        offset = random.randint(0, max_days_ahead - 1)

        return start + timedelta(days=offset)
    

    def call_llm(self, messages):
        # Mock some tool calls based on the user's question
        user_message = messages[-1]["content"].lower()
        tool_results = []

        if "cost" in user_message or "price" in user_message or "how much" in user_message:
            price = self.get_drug_price("MiraLAX")
            tool_results.append(f"Price: {price}")

        if "deliver" in user_message or "receive" in user_message:
            delivery_date = self.check_delivery_date("MiraLAX", "Lucy")
            tool_results.append(f"Delivery date: {delivery_date}")

        messages_for_llm = [
            # {
            #     "role": "system",
            #     "content": prompt.format(inputs=user_message),
            # },
            *messages,
        ]

        if tool_results:
            messages_for_llm.append(
                {"role": "system", "content": f"Tool results: {', '.join(tool_results)}"}
            )

        for chunk in self.client.chat.completions.create(
            model=self.model,
            messages=cast(Any, messages_for_llm),
            stream=True,
        ):
            yield chunk.to_dict()


    def predict(self, request: ResponsesAgentRequest):
        session_id = None
        if request.custom_inputs and "session_id" in request.custom_inputs:
            session_id = request.custom_inputs.get("session_id")
        elif request.context and request.context.conversation_id:
            session_id = request.context.conversation_id

        if session_id:
            mlflow.update_current_trace(
                metadata={
                    "mlflow.trace.session": session_id,
                }
            )

        outputs = [
            event.item
            for event in self.predict_stream(request)
            if event.type == "response.output_item.done"
        ]
        return ResponsesAgentResponse(
            output=outputs, custom_outputs=request.custom_inputs
        )

    def predict_stream(self, request: ResponsesAgentRequest):
        session_id = None
        if request.custom_inputs and "session_id" in request.custom_inputs:
            session_id = request.custom_inputs.get("session_id")
        elif request.context and request.context.conversation_id:
            session_id = request.context.conversation_id

        if session_id:
            mlflow.update_current_trace(
                metadata={
                    "mlflow.trace.session": session_id,
                }
            )
        
        messages = to_chat_completions_input([{"role": "system", "content": SYSTEM_PROMPT}]+[i.model_dump() for i in request.input])

        yield from output_to_responses_items_stream(self.call_llm(messages))


mlflow.openai.autolog()
agent = SimpleResponsesAgent(model="gpt-4o-mini")
mlflow.models.set_model(agent)
