import json
import os
from typing import Any, Dict, List, Optional, Union
import mlflow
import pandas as pd
from mlflow.models import set_model
from mlflow.models.rag_signatures import StringResponse, ChatCompletionRequest
from databricks.sdk import WorkspaceClient
from cookbook.agents.utils.chat import (
    get_messages_array,
    extract_user_query_string,
    extract_chat_history,
)
from cookbook.config.agents.story_builder_agent import StoryBuilderAgentConfig
from cookbook.agents.utils.load_config import load_config
import logging
import re

STORY_BUILDER_AGENT_DEFAULT_YAML_CONFIG_FILE_NAME = "story_builder_agent_config.yaml"


class StoryBuilderAgent(mlflow.pyfunc.PythonModel):
    """
    Class representing an Agent that creates data stories using the S.T.O.R.Y framework.
    """

    def __init__(
        self,
        agent_config: Optional[Union[StoryBuilderAgentConfig, str]] = None,
    ):
        super().__init__()
        self.model_serving_client = None
        self.chat_history = None
        self.agent_config = None

        # load the Agent's configuration
        self.agent_config = load_config(
            passed_agent_config=agent_config,
            default_config_file_name=STORY_BUILDER_AGENT_DEFAULT_YAML_CONFIG_FILE_NAME,
        )
        if not self.agent_config:
            logging.error(
                f"No agent config found. If you are in your local development environment, make sure you either [1] are calling init(agent_config=...) with either an instance of StoryBuilderAgentConfig or the full path to a YAML config file or [2] have a YAML config file saved at {{your_project_root_folder}}/configs/{STORY_BUILDER_AGENT_DEFAULT_YAML_CONFIG_FILE_NAME}."
            )
        else:
            logging.info("Successfully loaded agent config in __init__.")

            # Initialize the rest of the Agent
            w = WorkspaceClient(
                host="", # will put my host
                token="" # Will put my token
            )
            self.model_serving_client = w.serving_endpoints.get_open_ai_client()
            
            # Initialize the chat history to empty
            self.chat_history = []

    @mlflow.trace(name="agent", span_type="AGENT")
    def predict(
        self,
        context: Any = None,
        model_input: Union[ChatCompletionRequest, Dict, pd.DataFrame] = None,
        params: Any = None,
    ) -> StringResponse:
        # Check here to allow the Agent class to be initialized without a configuration file
        if not self.agent_config:
            raise RuntimeError("Agent config not loaded. Cannot call predict()")

        ##############################################################################
        # Extract `messages` key from the `model_input`
        messages = get_messages_array(model_input)

        ##############################################################################
        # Parse `messages` array into the user's query & the chat history
        with mlflow.start_span(name="parse_input", span_type="PARSER") as span:
            span.set_inputs({"messages": messages})
            last_message = extract_user_query_string(messages)
            last_message_role = messages[-1]["role"]
            self.chat_history = extract_chat_history(messages)
            span.set_outputs(
                {
                    "last_message": last_message,
                    "chat_history": self.chat_history,
                    "last_message_role": last_message_role,
                }
            )

        ##############################################################################
        # Identify data insights and visualization from previous messages (if any)
        with mlflow.start_span(name="extract_insights", span_type="PARSER") as span:
            span.set_inputs({"chat_history": self.chat_history, "last_message": last_message})
            
            # Extract data insights from the chat history (looking for Genie's output)
            data_insights = ""
            visualization_info = ""
            
            # Look for the latest message from assistant that contains data insights
            for msg in reversed(self.chat_history + [{"role": last_message_role, "content": last_message}]):
                # If there's a message from the Genie agent with data
                if msg["role"] == "assistant" and "name" in msg and msg["name"] == "Genie":
                    data_insights = msg["content"]
                    break
                # If there's a message that appears to be from Genie (contains a table)
                if msg["role"] == "assistant" and "|" in msg["content"] and "-|-" in msg["content"]:
                    data_insights = msg["content"]
                    break
                # If the message itself might contain a table
                if "|" in msg["content"] and "-|-" in msg["content"]:
                    data_insights = msg["content"]
                    break
            
            # Look for visualization information
            for msg in reversed(self.chat_history + [{"role": last_message_role, "content": last_message}]):
                # If there's a message from the Visualization agent
                if msg["role"] == "assistant" and "name" in msg and msg["name"] == "Visualization":
                    visualization_info = msg["content"]
                    break
                # If there's a message that appears to be a visualization (contains "Data Visualization")
                if msg["role"] == "assistant" and "Data Visualization" in msg["content"]:
                    visualization_info = msg["content"]
                    break
                
            span.set_outputs({
                "data_insights": data_insights[:500] + "..." if len(data_insights) > 500 else data_insights,
                "visualization_info": visualization_info[:500] + "..." if len(visualization_info) > 500 else visualization_info
            })

        ##############################################################################
        # Generate the STORY narrative using LLM
        with mlflow.start_span(name="create_story", span_type="LLM") as span:
            span.set_inputs({
                "data_insights": data_insights,
                "visualization_info": visualization_info,
                "last_message": last_message
            })
            
            # Create system prompt for story generation
            story_system_prompt = """You are a data storytelling expert. Your task is to create a compelling narrative using the S.T.O.R.Y framework:

S - Situation: Set the context by explaining the background and current state
T - Take off: Identify key trends, patterns, and changes in the data
O - Opportunity: Highlight insights and potential actions based on the data
R - Resolution: Provide specific recommendations or next steps
Y - Yield: Describe the potential benefits or outcomes from following the recommendations

Use the provided data insights and visualization to create a coherent and engaging story.
Format your response as a well-structured report with clear section headings.
"""

            # Create messages for the LLM
            story_messages = [
                {"role": "system", "content": story_system_prompt},
                {"role": "user", "content": f"""
Please create a data story using the following information:

DATA INSIGHTS:
{data_insights}

VISUALIZATION INFO:
{visualization_info}

USER QUERY:
{last_message}

Create a comprehensive S.T.O.R.Y narrative that explains the data, highlights key insights, and provides actionable recommendations.
"""}
            ]
            
            # Call the LLM to generate the story
            story_response = self.chat_completion(messages=story_messages)
            story_content = story_response.choices[0].message.content
            
            span.set_outputs({"story": story_content[:500] + "..." if len(story_content) > 500 else story_content})

        # Return the result
        return {
            "content": story_content,
            "messages": self.chat_history + [
                {"role": last_message_role, "content": last_message},
                {"role": "assistant", "content": story_content}
            ],
        }

    def chat_completion(self, messages: List[Dict[str, str]]):
        endpoint_name = self.agent_config.llm_endpoint_name
        
        # Trace the call to Model Serving
        traced_create = mlflow.trace(
            self.model_serving_client.chat.completions.create,
            name="chat_completions_api",
            span_type="CHAT_MODEL",
        )

        return traced_create(model=endpoint_name, messages=messages)


# tell MLflow logging where to find the agent's code
set_model(StoryBuilderAgent())