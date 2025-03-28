from typing import Any, List
from cookbook.config import SerializableConfig
from mlflow.models.resources import DatabricksResource, DatabricksGenieSpace


class GenieAgentConfig(SerializableConfig):
    """
    Configuration for the Genie agent with MLflow input example.

    Attributes:
        genie_space_id (str): ID of the Genie space to query.
        input_example (Any): Used by MLflow to set the Agent's input schema.
        encountered_error_user_message (str): Message to display when Genie encounters an error.
    """

    # Genie Space ID to query
    genie_space_id: str

    # Used by MLflow to set the Agent's input schema
    input_example: Any = {
        "messages": [
            {
                "role": "user",
                "content": "What types of data can I query?",
            },
        ]
    }

    # Error message to display when Genie encounters an error
    encountered_error_user_message: str = (
        "I encountered an error trying to answer your question, please try again."
    )

    def get_resource_dependencies(self) -> List[DatabricksResource]:
        return [DatabricksGenieSpace(genie_space_id=self.genie_space_id)]