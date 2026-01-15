import os

import mlflow
import openai

from openai import OpenAI

import pyspark.sql.functions as F

from datetime import date, timedelta
import random

from typing import Any, Dict, List, cast, Generator

from databricks.sdk import WorkspaceClient
from databricks_openai import UCFunctionToolkit, VectorSearchRetrieverTool
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
from unitycatalog.ai.core.base import get_uc_function_client

# Load system prompt
SYSTEM_PROMPT = mlflow.genai.load_prompt("prompts:/genai_apps.kpchat_mock.kpchat_system/6")

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
        for chunk in self.client.chat.completions.create(
            model=self.model,
            messages=messages,
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
        
        messages = to_chat_completions_input([{"role": "system", "content": SYSTEM_PROMPT.template}]+[i.model_dump() for i in request.input])

        yield from output_to_responses_items_stream(self.call_llm(messages))


mlflow.openai.autolog()
agent = SimpleResponsesAgent(model="gpt-4o-mini")
mlflow.models.set_model(agent)
