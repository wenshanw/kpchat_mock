import os

import mlflow
import openai

from openai import OpenAI

import pyspark.sql.functions as F

from datetime import date, timedelta
import random

from typing import Any, Dict, List, cast

mlflow.openai.autolog()

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


# Define your LLM endpoint
LLM_ENDPOINT_NAME = "gpt-4o-mini"

# Load prompt
SYSTEM_PROMPT = mlflow.genai.load_prompt("prompts:/rai.rai_prompts.kpchat_system/4")

