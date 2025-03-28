#/multi_agent_story/cookbook/config/agents/visualization_agent.py
from typing import List, Any, Dict
from cookbook.config import SerializableConfig, load_serializable_config_from_yaml
from mlflow.models.resources import DatabricksResource, DatabricksFunction
import yaml


class VisualizationAgentConfig(SerializableConfig):
    """
    Configuration for the Visualization agent.

    Attributes:
        llm_endpoint_name (str): Databricks Model Serving endpoint name.
        tools (List[Any]): Tools used by the visualization agent.
        input_example (Any): Used by MLflow to set the Agent's input schema.
        system_prompt (str): System prompt for the visualization agent.
    """

    llm_endpoint_name: str
    """Databricks Model Serving endpoint name."""

    system_prompt: str
    """System prompt template for the visualization agent."""
    
    tools: List[Any] = []
    """Tools used by the visualization agent."""

    input_example: Any = {
        "messages": [
            {
                "role": "user",
                "content": "Visualize this data: | Month | Revenue | Profit |\n|---|---|---|\n| Jan | 100 | 20 |\n| Feb | 110 | 22 |\n| Mar | 120 | 24 |",
            },
        ]
    }
    """Example input for the visualization agent."""

    def model_dump(self, **kwargs) -> Dict[str, Any]:
        """Override model_dump to serialize tools.

        Returns:
            Dict[str, Any]: Dictionary representation of the model.
        """
        model_dumped = super().model_dump(**kwargs)
        if self.tools:
            model_dumped["tools"] = [
                yaml.safe_load(tool.to_yaml()) if hasattr(tool, 'to_yaml') else tool for tool in self.tools
            ]
        return model_dumped

    @classmethod
    def _load_class_from_dict(
        cls, class_object, data: Dict[str, Any]
    ) -> "SerializableConfig":
        """Deserialize tools from the dictionary.

        Args:
            class_object: The class to instantiate.
            data: Dictionary containing serialized data.

        Returns:
            SerializableConfig: Instantiated class.
        """
        # Deserialize tools, dynamically reconstructing each tool
        if "tools" in data and data["tools"]:
            tools = []
            for tool_dict in data["tools"]:
                if isinstance(tool_dict, dict):
                    # Convert the tool dictionary back to a proper tool object
                    tool_yml = yaml.dump(tool_dict)
                    tools.append(load_serializable_config_from_yaml(tool_yml))
                else:
                    # The tool is already a proper object, just append it
                    tools.append(tool_dict)
            
            # Replace tools with deserialized instances
            data["tools"] = tools
        
        return class_object(**data)

    def get_resource_dependencies(self) -> List[DatabricksResource]:
        """Get all resource dependencies for this agent.

        Returns:
            List[DatabricksResource]: List of resource dependencies.
        """
        dependencies = [
            DatabricksResource(
                type="model-serving-endpoint", name=self.llm_endpoint_name
            )
        ]
        # Add resource dependencies from tools
        for tool in self.tools:
            if hasattr(tool, "get_resource_dependencies"):
                dependencies.extend(tool.get_resource_dependencies())
        
        return dependencies