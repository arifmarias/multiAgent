from typing import Any, List
from cookbook.config import SerializableConfig
from mlflow.models.resources import DatabricksResource


class StoryBuilderAgentConfig(SerializableConfig):
    """
    Configuration for the Story Builder agent.

    Attributes:
        llm_endpoint_name (str): Databricks Model Serving endpoint name.
        system_prompt (str): System prompt template for the story builder agent.
        input_example (Any): Used by MLflow to set the Agent's input schema.
    """

    llm_endpoint_name: str
    """Databricks Model Serving endpoint name."""

    system_prompt: str
    """System prompt template for the story builder agent."""

    input_example: Any = {
        "messages": [
            {
                "role": "user",
                "content": "Create a data story from this analysis: Revenue increased by 15% in Q2, with the East region showing the highest growth.",
            },
        ]
    }
    """Example input for the story builder agent."""
    
    def get_resource_dependencies(self) -> List[DatabricksResource]:
        """Get all resource dependencies for this agent.

        Returns:
            List[DatabricksResource]: List of resource dependencies.
        """
        return [
            DatabricksResource(
                type="model-serving-endpoint", name=self.llm_endpoint_name
            )
        ]