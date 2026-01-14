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


class SimpleResponsesAgent(ResponsesAgent):
    def __init__(self, model: str):
        self.client = OpenAI()
        self.model = model

    def call_llm(self, messages):
        for chunk in client.chat.completions.create(
            model=self.model,
            messages=messages,
            stream=True,
        ):
            yield chunk.to_dict()

    def predict(self, request: ResponsesAgentRequest):
        outputs = [
            event.item
            for event in self.predict_stream(request)
            if event.type == "response.output_item.done"
        ]
        return ResponsesAgentResponse(
            output=outputs, custom_outputs=request.custom_inputs
        )

    def predict_stream(self, request: ResponsesAgentRequest):
        messages = to_chat_completions_input([i.model_dump() for i in request.input])

        yield from output_to_responses_items_stream(self.call_llm(messages))


mlflow.openai.autolog()
agent = SimpleResponsesAgent()
mlflow.models.set_model(agent)
