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
from cookbook.agents.utils.execute_function import execute_function
from cookbook.config.agents.visualization_agent import VisualizationAgentConfig
from cookbook.agents.utils.load_config import load_config
import logging
import re

VISUALIZATION_AGENT_DEFAULT_YAML_CONFIG_FILE_NAME = "visualization_agent_config.yaml"


class VisualizationAgent(mlflow.pyfunc.PythonModel):
    """
    Class representing an Agent that creates visualizations from data.
    """

    def __init__(
        self,
        agent_config: Optional[Union[VisualizationAgentConfig, str]] = None,
    ):
        super().__init__()
        # Empty variables that will be initialized after loading the agent config.
        self.model_serving_client = None
        self.tool_functions = None
        self.tool_json_schemas = None
        self.chat_history = None
        self.agent_config = None

        # load the Agent's configuration. See load_config() for details.
        self.agent_config = load_config(
            passed_agent_config=agent_config,
            default_config_file_name=VISUALIZATION_AGENT_DEFAULT_YAML_CONFIG_FILE_NAME,
        )
        if not self.agent_config:
            logging.error(
                f"No agent config found. If you are in your local development environment, make sure you either [1] are calling init(agent_config=...) with either an instance of VisualizationAgentConfig or the full path to a YAML config file or [2] have a YAML config file saved at {{your_project_root_folder}}/configs/{VISUALIZATION_AGENT_DEFAULT_YAML_CONFIG_FILE_NAME}."
            )
        else:
            logging.info("Successfully loaded agent config in __init__.")

            # Now, initialize the rest of the Agent
            w = WorkspaceClient(
                host="", # Will Update Later 
                token="" # Will Update Later
            )
            self.model_serving_client = w.serving_endpoints.get_open_ai_client()

            # Initialize the tools
            self.tool_functions = {}
            self.tool_json_schemas = []
            for tool in self.agent_config.tools:
                self.tool_functions[tool.name] = tool
                self.tool_json_schemas.append(tool.get_json_schema())

            # Initialize the chat history to empty
            self.chat_history = []

    @mlflow.trace(name="agent", span_type="AGENT")
    def predict(
        self,
        context: Any = None,
        model_input: Union[ChatCompletionRequest, Dict, pd.DataFrame] = None,
        params: Any = None,
    ) -> StringResponse:
        # Check here to allow the Agent class to be initialized without a configuration file, which is required to import the class as a module in other files.
        if not self.agent_config:
            raise RuntimeError("Agent config not loaded. Cannot call predict()")

        ##############################################################################
        # Extract `messages` key from the `model_input`
        messages = get_messages_array(model_input)
        # Make sure messages is a list, not a string
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        ##############################################################################
        # Parse `messages` array into the user's query & the chat history
        with mlflow.start_span(name="parse_input", span_type="PARSER") as span:
            span.set_inputs({"messages": messages})
            # in a multi-agent setting, the last message can be from another assistant, not the user
            last_message = extract_user_query_string(messages)
            # Ensure we have a valid role
            if isinstance(messages, list) and len(messages) > 0 and isinstance(messages[-1], dict):
                last_message_role = messages[-1].get("role", "user")
            else:
                last_message_role = "user"
            # last_message_role = messages[-1]["role"]
            # Save the history inside the Agent's internal state
            self.chat_history = extract_chat_history(messages) if len(messages) > 1 else []
            span.set_outputs(
                {
                    "last_message": last_message,
                    "chat_history": self.chat_history,
                    "last_message_role": last_message_role,
                }
            )

        ##############################################################################
        # Call LLM to analyze data and create visualization

        # Step 1: Extract table from the last message
        with mlflow.start_span(name="extract_table", span_type="PARSER") as span:
            span.set_inputs({"last_message": last_message})
            # Look for a markdown table in the last message
            table_data = self.extract_table_from_message(last_message)
            span.set_outputs({"table_data": table_data})

        # Step 2: Generate data processing code
        with mlflow.start_span(name="generate_data_code", span_type="FUNCTION") as span:
            span.set_inputs({"table_data": table_data})
            
            # Import from our tools
            from tools.visualization_tools import extract_table_from_markdown, process_data_for_visualization
            
            if table_data:
                data_code = extract_table_from_markdown(table_data)
            else:
                # Try to extract JSON data from the entire message, not just a table format
                data_code = process_data_for_visualization(last_message)
                if data_code == "[]" or data_code == "null":
                    # Create fallback data for visualization
                    data_code = json.dumps([{"category": "Fallback", "value": 100}])
            
            span.set_outputs({"data_code": data_code})

        # Step 3: Determine best visualization type
        with mlflow.start_span(name="determine_visualization", span_type="LLM") as span:
            span.set_inputs({"table_data": table_data, "last_message": last_message})
            
            system_prompt = """You are a data visualization expert. Analyze the provided data and determine the best visualization type 
            (bar, line, scatter, pie, heatmap) based on the data structure and the user's query. 
            Respond with ONLY the visualization type and a brief explanation of why this type is appropriate.
            Format your response as: "TYPE: <visualization_type>\nREASON: <brief_explanation>"
            """
            
            if table_data:
                analysis_messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Data:\n{table_data}\n\nUser query: {last_message}"}
                ]
            else:
                analysis_messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"No table data found. User query: {last_message}"}
                ]
            
            # Call the LLM to determine the best visualization type
            analysis_response = self.chat_completion(messages=analysis_messages)
            analysis_text = analysis_response.choices[0].message.content
            
            # Extract visualization type from response
            viz_type_match = re.search(r"TYPE:\s*(\w+)", analysis_text, re.IGNORECASE)
            viz_type = viz_type_match.group(1).lower() if viz_type_match else "bar"
            
            # Extract reason from response
            reason_match = re.search(r"REASON:\s*(.+?)(?:$|\n)", analysis_text, re.IGNORECASE | re.DOTALL)
            reason = reason_match.group(1).strip() if reason_match else "This visualization type is appropriate for the data."
            
            span.set_outputs({"visualization_type": viz_type, "reason": reason})

        # Step 4: Generate and execute visualization code
        with mlflow.start_span(name="create_visualization", span_type="FUNCTION") as span:
            span.set_inputs({
                "data_code": data_code,
                "visualization_type": viz_type,
                "table_data": table_data
            })
            
            # Import from our tools
            from tools.visualization_tools import generate_visualization_code, python_viz_executor
            
            # Generate visualization code
            title = f"{viz_type.capitalize()} Chart of Data"
            viz_code = generate_visualization_code(data_code, viz_type, title)
            
            # Execute the code to generate visualization
            result_str = python_viz_executor(viz_code)
            
            try:
                # Try to parse the result
                result = json.loads(result_str)
            except json.JSONDecodeError:
                # Handle JSON parsing errors with a fallback result
                logging.error(f"Error parsing visualization result: {result_str}")
                result = {
                    "error": "Failed to parse visualization result",
                    "console_output": result_str
                }
            
            # Format the result for the response
            if "image" in result and result["image"]:
                image_data = result["image"]
                image_format = result.get("format", "image/png")
                html_img = f'<img src="data:{image_format};base64,{image_data}" alt="{title}" />'
                
                # Prepare the message
                visualization_output = f"""
        ## Data Visualization

        I've analyzed the data and created a {viz_type} chart:

        {html_img}

        ### Why a {viz_type.capitalize()} Chart?

        {reason}
        """
            else:
                # Create a fallback visualization
                import matplotlib.pyplot as plt
                import numpy as np
                import io
                import base64
                
                # Create a simple fallback chart
                plt.figure(figsize=(8, 5))
                plt.bar(['Fallback Data'], [100], color='lightblue')
                plt.title('Fallback Visualization')
                plt.ylabel('Value')
                
                # Convert to base64
                buffer = io.BytesIO()
                plt.savefig(buffer, format='png')
                buffer.seek(0)
                fallback_image = base64.b64encode(buffer.read()).decode('utf-8')
                plt.close()
                
                error_msg = result.get("error", "Failed to generate visualization")
                console_output = result.get("console_output", "")
                
                html_img = f'<img src="data:image/png;base64,{fallback_image}" alt="Fallback Visualization" />'
                
                visualization_output = f"""
        ## Data Visualization

        I attempted to create a visualization, but encountered an issue:

        **Error:** {error_msg}

        {html_img}

        This is a fallback visualization as I couldn't generate the requested chart.

        {console_output if console_output else ""}
        """
            
            span.set_outputs({"visualization_output": visualization_output})

    def create_fallback_visualization(title="Fallback Visualization", message="No data available"):
        """
        Creates a simple fallback visualization when regular visualization fails.
        
        Args:
            title (str): Title for the fallback visualization
            message (str): Message to display on the visualization
            
        Returns:
            str: Base64-encoded PNG image
        """
        try:
            import matplotlib.pyplot as plt
            import io
            import base64
            
            # Create a simple visualization
            plt.figure(figsize=(8, 5))
            
            # Either create a very simple bar chart
            plt.bar(['Fallback'], [100], color='lightblue')
            
            plt.title(title)
            plt.ylabel('Value')
            plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
            
            # Add the message
            plt.figtext(0.5, 0.3, message, wrap=True, 
                    horizontalalignment='center', fontsize=12)
            
            # Convert to base64
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png')
            buffer.seek(0)
            image_data = base64.b64encode(buffer.read()).decode('utf-8')
            plt.close()
            
            return f"data:image/png;base64,{image_data}"
        except Exception as e:
            return ""
    def extract_table_from_message(self, message: str) -> Optional[str]:
        """Extract table data from the message in various formats."""
        # First try to find a markdown table pattern
        table_pattern = r'(\|[^\n]*\|\n\|[-:| ]+\|\n(?:\|[^\n]*\|\n?)+)'
        table_match = re.search(table_pattern, message)
        
        if table_match:
            return table_match.group(1)
            
        # If no markdown table, look for JSON data
        try:
            # Look for JSON-like patterns
            json_pattern = r'\[\s*\{.*?\}\s*(?:,\s*\{.*?\}\s*)*\]'
            json_match = re.search(json_pattern, message, re.DOTALL)
            if json_match:
                # Validate it's proper JSON
                potential_json = json_match.group(0)
                json.loads(potential_json)  # This will raise an exception if not valid JSON
                return potential_json
        except json.JSONDecodeError:
            pass
            
        # If still no data found, check for key-value pairs
        # This is a fallback for when data isn't in a standard format
        kv_pattern = r'"([^"]+)"\s*:\s*([^,}\s]+|"[^"]+")'
        kv_matches = re.findall(kv_pattern, message)
        
        if kv_matches and len(kv_matches) >= 2:
            # Convert key-value pairs to a JSON object
            data_obj = {}
            for key, value in kv_matches:
                # Clean up and try to convert numeric values
                try:
                    clean_value = value.strip('"')
                    if '.' in clean_value and clean_value.replace('.', '', 1).isdigit():
                        data_obj[key] = float(clean_value)
                    elif clean_value.isdigit():
                        data_obj[key] = int(clean_value)
                    else:
                        data_obj[key] = clean_value
                except ValueError:
                    data_obj[key] = value
                    
            return json.dumps([data_obj])
            
        # If no structured data found, return None
        return None

    def chat_completion(self, messages: List[Dict[str, str]], tools: bool = False):
        endpoint_name = self.agent_config.llm_endpoint_name
        
        # Trace the call to Model Serving
        traced_create = mlflow.trace(
            self.model_serving_client.chat.completions.create,
            name="chat_completions_api",
            span_type="CHAT_MODEL",
        )

        if tools and self.tool_json_schemas:
            return traced_create(
                model=endpoint_name,
                messages=messages,
                tools=self.tool_json_schemas,
                parallel_tool_calls=False,
            )
        else:
            return traced_create(model=endpoint_name, messages=messages)


# tell MLflow logging where to find the agent's code
set_model(VisualizationAgent())