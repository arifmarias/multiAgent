from cookbook.config import _CLASS_PATH_KEY, serializable_config_to_yaml
from pydantic import BaseModel, field_validator
from typing import Any, List, Literal, Dict
from cookbook.config import (
    SerializableConfig,
)
from cookbook.config.shared.llm import LLMParametersConfig
from cookbook.config import (
    load_serializable_config_from_yaml,
)
import yaml
from mlflow.pyfunc import PythonModel
from typing import Optional


# Constants for the Multi-Agent Supervisor
FINISH_ROUTE_NAME = "FINISH"  # reserved name for the finish action
SUPERVISOR_ROUTE_NAME = "SUPERVISOR"  # reserved name for the supervisor agent
ROUTING_FUNCTION_NAME = "decide_next_worker_or_finish"  # function name for routing decision
WORKER_PROMPT_TEMPLATE = "<worker><name>{worker_name}</name><description>{worker_description}</description></worker>\n"
# Variable names for the supervisor's thinking process and decision making
CONVERSATION_HISTORY_THINKING_PARAM = "conversation_history_thinking"
WORKER_CAPABILITIES_THINKING_PARAM = "worker_capabilities_thinking"
NEXT_WORKER_OR_FINISH_PARAM = "next_worker_or_finish"

MULTI_AGENT_DEFAULT_YAML_CONFIG_FILE_NAME = "multi_agent_supervisor_config.yaml"


class SupervisedAgentConfig(SerializableConfig):
    """Configuration for a supervised agent within the multi-agent system.
    
    Attributes:
        name (str): Name of the supervised agent.
        description (str): Description of the agent's capabilities.
        endpoint_name (Optional[str]): Databricks Model Serving endpoint name (required for 'model_serving' mode).
        agent_config (Optional[SerializableConfig]): Agent's configuration (required for 'local' mode).
        agent_class_path (Optional[str]): Path to the agent's class (required for 'local' mode).
    """
    name: str
    description: str
    endpoint_name: Optional[str] = None
    agent_config: Optional[SerializableConfig] = None
    agent_class_path: Optional[str] = None

    def __init__(
        self,
        name: str,
        description: str,
        *,
        endpoint_name: Optional[str] = None,
        agent_config: Optional[SerializableConfig] = None,
        agent_class: Optional[type] = None,
        agent_class_path: Optional[str] = None,
    ):
        """Initialize a SupervisedAgentConfig instance.

        Args:
            name (str): Name of the supervised agent
            description (str): Description of the agent's capabilities
            endpoint_name (str): Databricks Model Serving endpoint name
            agent_config (Any): Agent's configuration
            agent_class (Any): Agent's implementation class
            agent_class_path (str): Path to the agent's class
        """
        if agent_class is not None and agent_class_path is not None:
            raise ValueError(
                "Only one of agent_class or agent_class_path can be provided"
            )

        if agent_class is not None:
            if not isinstance(agent_class, type):
                raise ValueError("agent_class must be an uninstantiated class")
            if not issubclass(agent_class, PythonModel):
                raise ValueError("agent_class must be a subclass of PythonModel")

            agent_class_path = f"{agent_class.__module__}.{agent_class.__name__}"

        if (endpoint_name is None) and (
            agent_config is None and agent_class_path is None
        ):
            raise ValueError(
                "One of endpoint_name or agent_config/agent_class(_path) must be provided"
            )

        super().__init__(
            name=name,
            description=description,
            endpoint_name=endpoint_name,
            agent_config=agent_config,
            agent_class_path=agent_class_path,
        )

    def model_dump(self, **kwargs) -> Dict[str, Any]:
        """Override model_dump to handle agent_config serialization.

        Returns:
            Dict[str, Any]: Dictionary representation of the model.
        """
        # only modify the method if agent_config is present, otherwise, this is not needed
        if self.agent_config is not None:
            kwargs["exclude"] = {"agent_config"}.union(kwargs.get("exclude", set()))
            model_dumped = super().model_dump(**kwargs)
            model_dumped["agent_config"] = yaml.safe_load(
                serializable_config_to_yaml(self.agent_config)
            )
            return model_dumped
        else:
            return super().model_dump(**kwargs)

    @classmethod
    def _load_class_from_dict(
        cls, class_object, data: Dict[str, Any]
    ) -> "SerializableConfig":
        # Deserialize agent config but only if it is present
        if data["agent_config"] is not None:
            agent_config = load_serializable_config_from_yaml(
                yaml.dump(data["agent_config"])
            )
            data["agent_config"] = agent_config

        return class_object(**data)


class MultiAgentSupervisorConfig(SerializableConfig):
    """
    Configuration for the multi-agent supervisor.

    Attributes:
        llm_endpoint_name (str): Databricks Model Serving endpoint name for the supervisor's LLM.
        llm_parameters (LLMParametersConfig): Parameters controlling LLM response behavior.
        input_example (Any): Example input used by MLflow to set the model's input schema.
        playground_debug_mode (bool): When True, outputs debug info to playground UI. Defaults to False.
        agent_loading_mode (str): Mode for loading supervised agents - "local" or "model_serving".
        max_supervisor_loops (int): Maximum number of supervisor cycles before finishing.
        supervisor_system_prompt (str): System prompt template for the supervisor agent.
        supervisor_user_prompt (str): User prompt template for the supervisor agent.
        supervisor_error_response (str): Response when an error occurs.
        agents (List[SupervisedAgentConfig]): List of supervised agents.
    """

    llm_endpoint_name: str
    """
    Databricks Model Serving endpoint name.
    This is the LLM used by the supervisor to make decisions.
    """

    llm_parameters: LLMParametersConfig
    """
    Parameters that control how the LLM responds, including temperature and max_tokens.
    """
    
    input_example: Any = {
        "messages": [
            {
                "role": "user",
                "content": "Analyze the monthly sales data and create a story about it.",
            },
        ]
    }
    """
    Example input used by MLflow to set the Agent's input schema.
    """

    playground_debug_mode: bool = False
    """
    Outputs details of all supervised agent's tool calling to the playground UI.
    """

    agent_loading_mode: Literal["local", "model_serving"] = "local"
    """
    Mode for loading supervised agents:
    - local: Supervised agent's code and config are loaded from your local environment.
    - model_serving: Supervised agent is deployed as a Model Serving endpoint.
    """

    @field_validator("max_supervisor_loops")
    def validate_max_loops(cls, v: int) -> int:
        if v <= 1:
            raise ValueError("max_supervisor_loops must be greater than 1")
        return v

    max_supervisor_loops: int = 5
    """
    The maximum turns of conversation before returning the last agent's response.
    """

    supervisor_system_prompt: str = """## Role
You are a supervisor responsible for managing a conversation between a user and the following workers. You select the next worker to respond or end the conversation to return the last worker's response to the user. Use the `{ROUTING_FUNCTION_NAME}` function to share your step-by-step reasoning and decision.

## Workers
{workers_names_and_descriptions}

## Objective
Your goal is to facilitate a three-step data analysis and storytelling process:
1. First, use the Genie agent to query and analyze the data
2. Next, use the Visualization agent to create an appropriate visualization of the data
3. Finally, use the Story Builder agent to create a narrative following the S.T.O.R.Y framework

## Instructions
1. **Review the Conversation History**: Think step by step to understand the user's request and the current state of the conversation. Output to the `{CONVERSATION_HISTORY_THINKING_PARAM}` variable.
2. **Assess Worker Descriptions**: Think step by step to consider each worker's capabilities in the context of the current conversation stage. Output to the `{WORKER_CAPABILITIES_THINKING_PARAM}` variable.
3. **Select the next worker OR finish the conversation**: Based on the conversation history and the worker's descriptions, decide which worker should respond next OR if the conversation should finish. Output either the  or "{FINISH_ROUTE_NAME}" to the `{NEXT_WORKER_OR_FINISH_PARAM}` variable.

## Flow Logic
- You MUST ALWAYS follow this EXACT sequence: Genie -> Visualization -> Story Builder -> FINISH
- You may ONLY select FINISH after ALL THREE AGENTS have completed their tasks
- You MUST select the next agent in the sequence regardless of the content of the previous agent's response
- If this is the first turn, ALWAYS select Genie
- If Genie has just responded, ALWAYS select Visualization next
- If Visualization has just responded, ALWAYS select StoryBuilder next
- ONLY select FINISH after StoryBuilder has responded
- Never select the same agent twice in a row
- IMPORTANT: You must check which agent responded last by looking at the 'name' field in the most recent assistant message
"""

    supervisor_user_prompt: str = (
        """Given the conversation history, the worker's descriptions and your thinking, which worker should act next OR should we FINISH? Respond with one of {worker_names_with_finish} to the `{NEXT_WORKER_OR_FINISH_PARAM}` variable in the `{ROUTING_FUNCTION_NAME}` function. If Genie agent already executed don't select genie agent again, Genie agent will be executed once, then visualization agent will be executed and after that story agent"""
    )
    """
    Prompt sent to supervisor after system prompt and conversation history to request next worker selection.
    """

    supervisor_error_response: str = "I'm sorry, but I encountered an error while processing your request. Please try again with a clearer question about data you'd like me to analyze and create a story about."

    agents: List[SupervisedAgentConfig]
    """
    List of supervised agents that will be called by the supervisor.
    """

    @classmethod
    def _load_class_from_dict(
        cls, class_object, data: Dict[str, Any]
    ) -> "SerializableConfig":
        # Deserialize agents, dynamically reconstructing each agent
        agents = []
        for agent_dict in data["agents"]:
            agent_yml = yaml.dump(agent_dict)
            agents.append(load_serializable_config_from_yaml(agent_yml))

        # Replace agents with deserialized instances
        data["agents"] = agents
        return class_object(**data)

    def model_dump(self, **kwargs) -> Dict[str, Any]:
        """Override model_dump to handle agent serialization.

        Returns:
            Dict[str, Any]: Dictionary representation of the model.
        """

        model_dumped = super().model_dump(**kwargs)
        model_dumped["agents"] = [
            yaml.safe_load(serializable_config_to_yaml(agent)) for agent in self.agents
        ]
        return model_dumped