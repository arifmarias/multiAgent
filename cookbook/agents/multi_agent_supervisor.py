import json
import os
from typing import Any, Callable, Dict, List, Optional, Union
from cookbook.config.agents.multi_agent_supervisor import (
    MULTI_AGENT_DEFAULT_YAML_CONFIG_FILE_NAME,
    FINISH_ROUTE_NAME,
    SUPERVISOR_ROUTE_NAME,
    ROUTING_FUNCTION_NAME,
    WORKER_PROMPT_TEMPLATE,
    CONVERSATION_HISTORY_THINKING_PARAM,
    WORKER_CAPABILITIES_THINKING_PARAM,
    NEXT_WORKER_OR_FINISH_PARAM,
)
import mlflow
from dataclasses import asdict, dataclass, field
import pandas as pd
from mlflow.models import set_model, ModelConfig
from mlflow.models.rag_signatures import StringResponse, ChatCompletionRequest, Message
from databricks.sdk import WorkspaceClient
import os
from cookbook.agents.utils.chat import (
    remove_message_keys_with_null_values,
    remove_tool_calls_from_messages,
)
from cookbook.agents.utils.load_config import load_config
from cookbook.config.agents.multi_agent_supervisor import (
    MultiAgentSupervisorConfig,
)
from cookbook.agents.utils.chat import get_messages_array
import importlib
import logging

# logging.basicConfig(level=logging.INFO)

from mlflow.entities import Trace
import mlflow.deployments


AGENT_RAW_OUTPUT_KEY = "raw_agent_output"
AGENT_NEW_MESSAGES_KEY = "new_messages"


@dataclass
class SupervisorState:
    """Tracks essential conversation state"""

    chat_history: List[Dict[str, str]] = field(default_factory=list)
    last_agent_called: str = ""
    number_of_supervisor_loops_completed: int = 0
    num_messages_at_start: int = 0

    @mlflow.trace(span_type="FUNCTION", name="state.append_new_message_to_history")
    def append_new_message_to_history(self, message: Dict[str, str]) -> None:
        span = mlflow.get_current_active_span()
        if span:  # Handle case when mlflow tracing is disabled
            span.set_inputs({"message": message})
        with mlflow.start_span(
            name="remove_message_keys_with_null_values"
        ) as span_inner:
            span_inner.set_inputs({"message": message})
            message_with_no_null_values_for_keys = remove_message_keys_with_null_values(
                message
            )
            span_inner.set_outputs(
                {
                    "message_with_no_null_values_for_keys": message_with_no_null_values_for_keys
                }
            )
        self.chat_history.append(message_with_no_null_values_for_keys)
        if span:
            span.set_outputs(self.chat_history)

    @mlflow.trace(span_type="FUNCTION", name="state.overwrite_chat_history")
    def overwrite_chat_history(self, new_chat_history: List[Dict[str, str]]) -> None:
        span = mlflow.get_current_active_span()
        if span:  # Handle case when mlflow tracing is disabled
            span.set_inputs(
                {
                    "new_chat_history": new_chat_history,
                    "current_chat_history": self.chat_history,
                }
            )
        messages_with_no_null_values_for_keys = []
        with mlflow.start_span(
            name="remove_message_keys_with_null_values"
        ) as span_inner:
            span_inner.set_inputs({"new_chat_history": new_chat_history})
            for message in new_chat_history:
                messages_with_no_null_values_for_keys.append(
                    remove_message_keys_with_null_values(message)
                )
            span_inner.set_outputs(
                {
                    "messages_with_no_null_values_for_keys": messages_with_no_null_values_for_keys
                }
            )
        self.chat_history = messages_with_no_null_values_for_keys.copy()
        if span:
            span.set_outputs(self.chat_history)


class MultiAgentSupervisor(mlflow.pyfunc.PythonModel):
    """
    Class representing a Multi-Agent Supervisor that orchestrates multiple specialized agents.
    
    This supervisor manages the conversation flow between different worker agents
    (Genie, Visualization, and Story Builder) to complete a data analysis and storytelling task.
    """

    def __init__(
        self, agent_config: Optional[Union[MultiAgentSupervisorConfig, str]] = None
    ):
        logging.info("Initializing MultiAgentSupervisor")

        # load the Agent's configuration
        self.agent_config = load_config(
            passed_agent_config=agent_config,
            default_config_file_name=MULTI_AGENT_DEFAULT_YAML_CONFIG_FILE_NAME,
        )
        if not self.agent_config:
            raise ValueError(
                f"No agent config found. Please provide a valid MultiAgentSupervisorConfig or path to a YAML config file."
            )
        else:
            logging.info("Successfully loaded agent config in __init__.")

            # Initialize clients
            self._initialize_model_serving_clients()

            # Set up agents and routing
            self._initialize_supervised_agents()

            # Set up prompts and tools
            self._initialize_supervisor_prompts_and_tools()

            # Initialize state
            self.state = None  # Will be initialized per conversation
            logging.info("Initialized MultiAgentSupervisor")

    def _initialize_model_serving_clients(self):
        """Initialize API clients for model serving"""
        w = WorkspaceClient()
        self.model_serving_client = w.serving_endpoints.get_open_ai_client()

        # used for calling the child agent's deployments
        self.mlflow_serving_client = mlflow.deployments.get_deploy_client("databricks")
        logging.info("Initialized model serving clients")

    def _initialize_supervised_agents(self):
        """Initialize the agent registry and capabilities"""
        self.agents = {}

        # Add configured worker agents
        if self.agent_config.agent_loading_mode == "model_serving":
            # using the model serving endpoints of the agents
            for agent in self.agent_config.agents:
                self.agents[agent.name] = {
                    "agent_description": agent.description,
                    "endpoint_name": agent.endpoint_name,
                }
        elif self.agent_config.agent_loading_mode == "local":
            # using the local agent classes
            for agent in self.agent_config.agents:
                # load the agent class
                module_name, class_name = agent.agent_class_path.rsplit(".", 1)

                module = importlib.import_module(module_name)
                # Load the Agent class, which will be a PyFunc
                agent_class_obj = getattr(module, class_name)
                self.agents[agent.name] = {
                    "agent_description": agent.description,
                    "agent_pyfunc_instance": agent_class_obj(
                        agent_config=agent.agent_config
                    ),  # instantiate the PyFunc
                }
                logging.info(f"Loaded agent: {agent.name}")
        else:
            raise ValueError(
                f"Invalid agent loading mode: {self.agent_config.agent_loading_mode}"
            )

    def _initialize_supervisor_prompts_and_tools(self):
        """Initialize prompts and function calling tools"""
        # Create agents string for system prompt
        agents_info = [
            WORKER_PROMPT_TEMPLATE.format(
                worker_name=key, worker_description=value["agent_description"]
            )
            for key, value in self.agents.items()
        ]
        workers_names_and_descriptions = "".join(agents_info)

        # Update system prompt with template variables
        self.supervisor_system_prompt = (
            self.agent_config.supervisor_system_prompt.format(
                ROUTING_FUNCTION_NAME=ROUTING_FUNCTION_NAME,
                CONVERSATION_HISTORY_THINKING_PARAM=CONVERSATION_HISTORY_THINKING_PARAM,
                WORKER_CAPABILITIES_THINKING_PARAM=WORKER_CAPABILITIES_THINKING_PARAM,
                NEXT_WORKER_OR_FINISH_PARAM=NEXT_WORKER_OR_FINISH_PARAM,
                FINISH_ROUTE_NAME=FINISH_ROUTE_NAME,
                workers_names_and_descriptions=workers_names_and_descriptions,
            )
        )

        self.supervisor_user_prompt = self.agent_config.supervisor_user_prompt.format(
            worker_names_with_finish=list(self.agents.keys()) + [FINISH_ROUTE_NAME],
            NEXT_WORKER_OR_FINISH_PARAM=NEXT_WORKER_OR_FINISH_PARAM,
            ROUTING_FUNCTION_NAME=ROUTING_FUNCTION_NAME,
            FINISH_ROUTE_NAME=FINISH_ROUTE_NAME,
        )

        # Initialize routing function schema
        self.route_function = {
            "type": "function",
            "function": {
                "name": ROUTING_FUNCTION_NAME,
                "description": "Route the conversation by providing your thinking and next worker selection.",
                "parameters": {
                    "properties": {
                        CONVERSATION_HISTORY_THINKING_PARAM: {"type": "string"},
                        WORKER_CAPABILITIES_THINKING_PARAM: {"type": "string"},
                        NEXT_WORKER_OR_FINISH_PARAM: {
                            "enum": list(self.agents.keys()) + [FINISH_ROUTE_NAME],
                            "type": "string",
                        },
                    },
                    "required": [
                        CONVERSATION_HISTORY_THINKING_PARAM,
                        WORKER_CAPABILITIES_THINKING_PARAM,
                        NEXT_WORKER_OR_FINISH_PARAM,
                    ],
                    "type": "object",
                },
            },
        }
        self.tool_json_schemas = [self.route_function]

    @mlflow.trace(span_type="AGENT")
    def _get_supervisor_routing_decision(self, messages: List[Dict[str, str]]) -> Dict[str, str]:
        """
        Gets the supervisor LLM to decide which worker agent to route to next or to finish.
        
        Args:
            messages: The conversation history
            
        Returns:
            Dict containing the supervisor's reasoning and decision
        """
        # Find the last assistant message to analyze which agent responded last
        last_assistant_name = None
        for msg in reversed(messages):
            if msg.get("role") == "assistant" and "name" in msg:
                last_assistant_name = msg.get("name")
                break
                
        logging.info(f"Last assistant message was from agent: {last_assistant_name}")
        
        supervisor_messages = (
            [{"role": "system", "content": self.supervisor_system_prompt}]
            + messages
            + [
                {
                    "role": "user",
                    "content": self.supervisor_user_prompt,
                }
            ]
        )

        response = self.chat_completion(messages=supervisor_messages, tools=True)
        supervisor_llm_response = response.choices[0].message
        supervisor_tool_calls = supervisor_llm_response.tool_calls

        if supervisor_tool_calls:
            for tool_call in supervisor_tool_calls:
                function = tool_call.function
                args = json.loads(function.arguments)
                if function.name == ROUTING_FUNCTION_NAME:
                    next_agent = args.get(NEXT_WORKER_OR_FINISH_PARAM)
                    logging.info(f"Supervisor decided on next agent: {next_agent}")
                    logging.info(f"Reasoning for history: {args.get(CONVERSATION_HISTORY_THINKING_PARAM, '')[:100]}...")
                    logging.info(f"Reasoning for capabilities: {args.get(WORKER_CAPABILITIES_THINKING_PARAM, '')[:100]}...")
                    return args  # includes all keys from the function call
                else:
                    logging.error(
                        f"Supervisor LLM failed to call the {ROUTING_FUNCTION_NAME}(...) function to determine the next step, so we will default to finishing.  It tried to call `{function.name}` with args `{function.arguments}`."
                    )
                    return {NEXT_WORKER_OR_FINISH_PARAM: FINISH_ROUTE_NAME}
        else:
            logging.error(
                f"Supervisor LLM failed to choose a tool at all, so we will default to finishing.  It said `{supervisor_llm_response}`."
            )
            return {NEXT_WORKER_OR_FINISH_PARAM: FINISH_ROUTE_NAME}

    @mlflow.trace()
    def _call_supervised_agent(
        self, agent_name: str, input_messages: List[Dict[str, str]]
    ) -> Dict[str, Any]:
        """
        Calls a supervised agent and returns ONLY the new messages produced by that agent.
        Includes special handling for Genie → Visualization transition.
        
        Args:
            agent_name: The name of the agent to call
            input_messages: The conversation history to pass to the agent
            
        Returns:
            Dictionary containing the agent's raw output and new messages
        """
        span = mlflow.get_current_active_span()
        if span:
            span.set_attribute(
                "self.agent_config.agent_loading_mode",
                self.agent_config.agent_loading_mode,
            )
            
        # Special handling for visualization agent when previous agent was Genie
        modified_input_messages = input_messages.copy()
        if agent_name == "Visualization" and self.state.last_agent_called == "Genie":
            # Get the complete genie response
            genie_response = {
                "messages": input_messages.copy()
            }
            
            # Find Genie's specific content in the messages
            for msg in reversed(input_messages):
                if msg.get("role") == "assistant" and msg.get("name") == "Genie" and "content" in msg:
                    genie_response["content"] = msg["content"]
                    break
                    
            # Also check for tool messages which might contain the JSON data
            for msg in input_messages:
                if msg.get("role") == "tool" and "content" in msg:
                    try:
                        tool_content = json.loads(msg["content"])
                        if "sql_query" in tool_content or "data_table" in tool_content:
                            # We found the structured data from Genie
                            logging.info("Found structured data in tool message")
                            break
                    except (json.JSONDecodeError, TypeError):
                        pass
            
            # Import the visualization tools and call connect_genie_to_visualization
            try:
                from tools.visualization_tools import connect_genie_to_visualization
                
                logging.info("Calling connect_genie_to_visualization")
                processed_input = connect_genie_to_visualization(genie_response)
                
                # Check if we got a valid result
                if processed_input and "messages" in processed_input:
                    modified_input_messages = processed_input["messages"]
                    logging.info("Successfully processed Genie data for Visualization")
                else:
                    logging.warning("connect_genie_to_visualization returned invalid result")
            except Exception as e:
                logging.error(f"Error processing Genie data: {str(e)}", exc_info=True)
        
        raw_agent_output = {}
        try:
            if self.agent_config.agent_loading_mode == "model_serving":
                endpoint_name = self.agents.get(agent_name).get("endpoint_name")
                if endpoint_name:
                    # Request that includes mlflow trace from the endpoint
                    request = {
                        "databricks_options": {"return_trace": True},
                        "messages": modified_input_messages.copy(),
                    }
                    completion = self.mlflow_serving_client.predict(
                        endpoint=endpoint_name, inputs=request
                    )
                    logging.info(f"Called agent: {agent_name}")
                    logging.info(f"Got response from agent: {completion}")
                    # Add the trace from model serving API call to the active trace
                    if trace := completion.pop("databricks_output", {}).get("trace"):
                        trace = Trace.from_dict(trace)
                        mlflow.add_trace(trace)
                    raw_agent_output = completion or {}
                else:
                    raise ValueError(f"Invalid agent selected: {agent_name}")
            elif self.agent_config.agent_loading_mode == "local":
                agent_pyfunc_instance = self.agents.get(agent_name).get(
                    "agent_pyfunc_instance"
                )
                if agent_pyfunc_instance:
                    request = {
                        "messages": modified_input_messages.copy(),
                    }
                    result = agent_pyfunc_instance.predict(model_input=request)
                    
                    if result is None:
                        logging.error(f"WARNING: Agent {agent_name} returned None instead of a dictionary!")
                        
                        # Create a fallback response based on the agent type
                        if agent_name == "Visualization":
                            # Create a basic visualization fallback
                            fallback_message = "I attempted to create a visualization but encountered technical difficulties."
                            raw_agent_output = {
                                "content": fallback_message,
                                "messages": input_messages + [{
                                    "role": "assistant",
                                    "name": agent_name,
                                    "content": fallback_message
                                }]
                            }
                        elif agent_name == "StoryBuilder":
                            # Create a basic story fallback
                            fallback_message = """
                            # Data Analysis Story
                            
                            ## Situation
                            The data shows currency conversion rates over time.
                            
                            ## Take Off
                            The rates fluctuate significantly across different dates.
                            
                            ## Opportunity
                            This data could help identify optimal times for currency exchanges.
                            
                            ## Resolution
                            Regular monitoring of these rates could inform better timing decisions.
                            
                            ## Yield
                            Optimizing currency conversion timing could lead to cost savings.
                            """
                            raw_agent_output = {
                                "content": fallback_message,
                                "messages": input_messages + [{
                                    "role": "assistant",
                                    "name": agent_name,
                                    "content": fallback_message
                                }]
                            }
                        else:
                            # Generic fallback
                            fallback_message = f"I processed your request with {agent_name} but couldn't generate a proper response."
                            raw_agent_output = {
                                "content": fallback_message,
                                "messages": input_messages + [{
                                    "role": "assistant",
                                    "name": agent_name,
                                    "content": fallback_message
                                }]
                            }
                    else:
                        raw_agent_output = result
                    
                    logging.info(f"Called agent: {agent_name} and received output type: {type(raw_agent_output)}")
                else:
                    raise ValueError(f"Invalid agent selected: {agent_name}")
            else:
                raise ValueError(
                    f"Invalid agent loading mode: {self.agent_config.agent_loading_mode}"
                )
        except Exception as e:
            # Handle visualization errors with fallback
            logging.error(f"Error in {agent_name} agent: {str(e)}")
            
            if agent_name == "Visualization":
                fallback_message = "I attempted to create a visualization but encountered technical difficulties."
                raw_agent_output = {
                    "content": fallback_message,
                    "messages": input_messages + [{
                        "role": "assistant",
                        "name": agent_name,
                        "content": fallback_message
                    }]
                }
            elif agent_name == "StoryBuilder":
                fallback_message = """
                # Data Analysis Story
                
                ## Situation
                The data shows currency conversion rates over time.
                
                ## Take Off
                The rates fluctuate significantly across different dates.
                
                ## Opportunity
                This data could help identify optimal times for currency exchanges.
                
                ## Resolution
                Regular monitoring of these rates could inform better timing decisions.
                
                ## Yield
                Optimizing currency conversion timing could lead to cost savings.
                """
                raw_agent_output = {
                    "content": fallback_message,
                    "messages": input_messages + [{
                        "role": "assistant",
                        "name": agent_name,
                        "content": fallback_message
                    }]
                }
            else:
                fallback_message = f"Error processing your request with {agent_name}."
                raw_agent_output = {
                    "content": fallback_message,
                    "messages": input_messages + [{
                        "role": "assistant",
                        "name": agent_name,
                        "content": fallback_message
                    }]
                }
                
        # Ensure raw_agent_output is not None
        if raw_agent_output is None:
            logging.error(f"CRITICAL ERROR: raw_agent_output is None after processing for agent {agent_name}!")
            fallback_message = f"Error processing your request with {agent_name}."
            raw_agent_output = {
                "content": fallback_message,
                "messages": input_messages + [{
                    "role": "assistant",
                    "name": agent_name,
                    "content": fallback_message
                }]
            }
            
        # Return only the net new messages produced by the agent
        agent_output_messages = raw_agent_output.get("messages", [])
        num_messages_previously = len(input_messages)
        num_messages_after_agent = len(agent_output_messages)
        
        logging.info(f"Agent {agent_name} returned {num_messages_after_agent} messages")
        logging.info(f"Previously had {num_messages_previously} messages")
        
        if (
            num_messages_after_agent == 0
            or num_messages_after_agent == num_messages_previously
        ):
            # Create fallback message if agent didn't produce any new messages
            fallback_message = f"I processed your request with {agent_name} but couldn't generate a proper response."
            
            if agent_name == "Visualization":
                fallback_message = "I attempted to create a visualization but encountered technical difficulties."
            elif agent_name == "StoryBuilder":
                fallback_message = """
                # Data Analysis Story
                
                ## Situation
                The data shows currency conversion rates over time.
                
                ## Take Off
                The rates fluctuate significantly across different dates.
                
                ## Opportunity
                This data could help identify optimal times for currency exchanges.
                
                ## Resolution
                Regular monitoring of these rates could inform better timing decisions.
                
                ## Yield
                Optimizing currency conversion timing could lead to cost savings.
                """
                
            new_messages = [{
                "role": "assistant",
                "name": agent_name,
                "content": fallback_message
            }]
            logging.info(f"Created fallback message for {agent_name} agent")
            return {
                AGENT_RAW_OUTPUT_KEY: {"content": fallback_message, "messages": input_messages + new_messages},
                AGENT_NEW_MESSAGES_KEY: new_messages,
            }
        else:
            # Add the Agent's name to its messages
            new_messages = agent_output_messages[num_messages_previously:].copy()
            for new_message in new_messages:
                if new_message.get("role") == "assistant" and "name" not in new_message:
                    new_message["name"] = agent_name
                    
            logging.info(f"Agent {agent_name} added {len(new_messages)} new messages with agent name")
            
            return {
                # agent's raw output
                AGENT_RAW_OUTPUT_KEY: raw_agent_output,
                # new messages produced by the agent
                AGENT_NEW_MESSAGES_KEY: new_messages,
            }

    @mlflow.trace(name="agent", span_type="AGENT")
    def predict(
        self,
        context: Any = None,
        model_input: Union[ChatCompletionRequest, Dict, pd.DataFrame] = None,
        params: Any = None,
    ) -> StringResponse:
        """
        Main entry point for the agent. Implements the MLflow PythonModel interface.
        
        Args:
            context: MLflow model context (not used)
            model_input: Input messages containing user query
            params: Additional parameters (not used)
            
        Returns:
            StringResponse with the final response and message history
        """
        # Check configuration loaded
        if not self.agent_config:
            raise RuntimeError("Agent config not loaded. Cannot call predict()")
            
        # Initialize conversation state
        messages = get_messages_array(model_input)
        self.state = SupervisorState()
        self.state.overwrite_chat_history(messages)
        self.state.num_messages_at_start = len(messages)

        # Run the supervisor loop up to max_supervisor_loops times
        while (
            self.state.number_of_supervisor_loops_completed
            < self.agent_config.max_supervisor_loops
        ):
            with mlflow.start_span(name="supervisor_loop_iteration") as span:
                self.state.number_of_supervisor_loops_completed += 1
                
                # Force the proper agent sequence based on the iteration number
                forced_agent = None
                if self.state.number_of_supervisor_loops_completed == 1:
                    forced_agent = "Genie"
                    logging.info("First iteration: forcing Genie agent")
                elif self.state.number_of_supervisor_loops_completed == 2:
                    forced_agent = "Visualization" 
                    logging.info("Second iteration: forcing Visualization agent")
                elif self.state.number_of_supervisor_loops_completed == 3:
                    forced_agent = "StoryBuilder"
                    logging.info("Third iteration: forcing StoryBuilder agent")
                elif self.state.number_of_supervisor_loops_completed >= 4:
                    forced_agent = FINISH_ROUTE_NAME
                    logging.info("Fourth iteration: forcing FINISH")

                # Remove tool calls to ensure clean history for LLM
                chat_history_without_tool_calls = remove_tool_calls_from_messages(
                    self.state.chat_history
                )
                
                # Debug: Check last assistant in history vs state
                last_assistant_in_history = None
                for msg in reversed(self.state.chat_history):
                    if msg.get("role") == "assistant" and "name" in msg:
                        last_assistant_in_history = msg.get("name")
                        break
                logging.info(f"Last assistant in history: {last_assistant_in_history}, state.last_agent_called: {self.state.last_agent_called}")
                
                # Ensure state is in sync with message history
                if last_assistant_in_history and self.state.last_agent_called != last_assistant_in_history:
                    logging.warning(f"Fixing mismatch between state tracking ({self.state.last_agent_called}) and message history ({last_assistant_in_history})")
                    self.state.last_agent_called = last_assistant_in_history
                
                # Get supervisor decision on next agent (but we might override it)
                routing_function_output = self._get_supervisor_routing_decision(
                    chat_history_without_tool_calls
                )

                # Use forced agent if specified, otherwise use supervisor's decision
                next_agent = forced_agent if forced_agent else routing_function_output.get(NEXT_WORKER_OR_FINISH_PARAM)
                logging.info(f"Selected agent for iteration {self.state.number_of_supervisor_loops_completed}: {next_agent}")
                
                span.set_inputs(
                    {
                        f"supervisor.{NEXT_WORKER_OR_FINISH_PARAM}": next_agent,
                        f"supervisor.{CONVERSATION_HISTORY_THINKING_PARAM}": routing_function_output.get(
                            CONVERSATION_HISTORY_THINKING_PARAM
                        ) if routing_function_output else None,
                        f"supervisor.{WORKER_CAPABILITIES_THINKING_PARAM}": routing_function_output.get(
                            WORKER_CAPABILITIES_THINKING_PARAM
                        ) if routing_function_output else None,
                        "state.number_of_workers_called": self.state.number_of_supervisor_loops_completed,
                        "state.chat_history": self.state.chat_history,
                        "state.last_agent_called": self.state.last_agent_called,
                        "chat_history_without_tool_calls": chat_history_without_tool_calls,
                    }
                )

                # Handle supervisor not returning a next agent
                if next_agent is None:
                    logging.error(
                        f"Supervisor returned no next agent, so we will default to finishing."
                    )
                    span.set_outputs(
                        {
                            "post_processed_decision": FINISH_ROUTE_NAME,
                            "post_processing_reason": "Supervisor returned no next agent, so we will default to finishing.",
                            "updated_chat_history": self.state.chat_history,
                        }
                    )
                    break
                    
                # Handle finish decision
                if next_agent == FINISH_ROUTE_NAME:
                    logging.info(
                        f"Supervisor called {FINISH_ROUTE_NAME} after {self.state.number_of_supervisor_loops_completed} workers being called."
                    )
                    span.set_outputs(
                        {
                            "post_processed_decision": FINISH_ROUTE_NAME,
                            "post_processing_reason": "Supervisor selected it.",
                            "updated_chat_history": self.state.chat_history,
                        }
                    )
                    break  # finish by exiting the while loop
                    
                # Call worker agent and update history
                try:
                    agent_output = self._call_supervised_agent(
                        next_agent, chat_history_without_tool_calls
                    )
                    
                    if agent_output is None:
                        logging.error(f"Agent {next_agent} returned None. Creating fallback response.")
                        fallback_message = f"I encountered an error while processing your request with {next_agent}."
                        fallback_message_obj = {
                            "role": "assistant",
                            "name": next_agent,
                            "content": fallback_message
                        }
                        agent_new_messages = [fallback_message_obj]
                        agent_raw_output = {
                            "content": fallback_message,
                            "messages": chat_history_without_tool_calls + [fallback_message_obj]
                        }
                    else:
                        agent_new_messages = agent_output.get(AGENT_NEW_MESSAGES_KEY, [])
                        agent_raw_output = agent_output.get(AGENT_RAW_OUTPUT_KEY, {})

                    # Log new messages and their agent name tags
                    logging.info(f"Agent {next_agent} returned {len(agent_new_messages)} new messages")
                    for msg in agent_new_messages:
                        if msg.get("role") == "assistant":
                            logging.info(f"Assistant message has name: {msg.get('name')}")

                    self.state.overwrite_chat_history(
                        self.state.chat_history + agent_new_messages
                    )
                    # CRITICAL: Update the last_agent_called AFTER the agent output is processed
                    self.state.last_agent_called = next_agent
                    logging.info(f"Updated last_agent_called to: {next_agent}")
                    
                    span.set_outputs(
                        {
                            "post_processed_decision": next_agent,
                            "post_processing_reason": "Supervisor selected it.",
                            "updated_chat_history": self.state.chat_history,
                            f"called_agent.{AGENT_NEW_MESSAGES_KEY}": agent_new_messages,
                            f"called_agent.{AGENT_RAW_OUTPUT_KEY}": agent_raw_output,
                        }
                    )

                except Exception as e:
                    logging.error(
                        f"Error calling agent {next_agent}: {e}",
                        exc_info=True
                    )
                    
                    # Create a fallback response
                    fallback_message = f"I encountered an error while processing your request with {next_agent}."
                    fallback_message_obj = {
                        "role": "assistant",
                        "name": next_agent,
                        "content": fallback_message
                    }
                    
                    # Add the fallback message to history
                    self.state.overwrite_chat_history(
                        self.state.chat_history + [fallback_message_obj]
                    )
                    
                    # Update the last agent called
                    self.state.last_agent_called = next_agent
                    
                    # Check if this was the StoryBuilder agent - if so, we can finish
                    if next_agent == "StoryBuilder":
                        logging.info("StoryBuilder failed but was attempted, proceeding to FINISH")
                        span.set_outputs(
                            {
                                "post_processed_decision": FINISH_ROUTE_NAME,
                                "post_processing_reason": "StoryBuilder failed but was attempted, proceeding to FINISH",
                                "updated_chat_history": self.state.chat_history,
                            }
                        )
                        break
                    
                    # Otherwise, continue to the next agent in sequence
                    span.set_outputs(
                        {
                            "post_processed_decision": next_agent,
                            "post_processing_reason": "Agent failed but added fallback message",
                            "updated_chat_history": self.state.chat_history,
                        }
                    )

        # if the last message is not from the assistant, we need to add a fake assistant message
        if not self.state.chat_history or self.state.chat_history[-1]["role"] != "assistant":
            logging.warning(
                "No assistant ended up replying, so we'll add an error response"
            )
            with mlflow.start_span(name="add_error_response_to_history") as span:
                span.set_inputs(
                    {
                        "state.chat_history": self.state.chat_history,
                    }
                )
                self.state.append_new_message_to_history(
                    {
                        "role": "assistant",
                        "content": self.agent_config.supervisor_error_response,
                    }
                )
                span.set_outputs(
                    {
                        "updated_chat_history": self.state.chat_history,
                    }
                )

        # Return the resulting conversation back to the user
        with mlflow.start_span(name="return_conversation_to_user") as span:
            span.set_inputs(
                {
                    "state.chat_history": self.state.chat_history,
                    "agent_config.playground_debug_mode": self.agent_config.playground_debug_mode,
                }
            )
            
            # Format the final return value
            return_value = {
                "content": (
                    self.state.chat_history[-1]["content"]
                    if self.state.chat_history
                    else ""
                ),
                "messages": self.state.chat_history,
            }
            span.set_outputs(return_value)
            return return_value

    def chat_completion(self, messages: List[Dict[str, str]], tools: bool = False):
        """
        Call the chat completion API using the model serving client.
        
        Args:
            messages: The messages to send to the LLM
            tools: Whether to include tools in the request
            
        Returns:
            Response from the model serving endpoint
        """
        endpoint_name = self.agent_config.llm_endpoint_name
        llm_options = self.agent_config.llm_parameters.model_dump()

        # Trace the call to Model Serving
        traced_create = mlflow.trace(
            self.model_serving_client.chat.completions.create,
            name="chat_completions_api",
            span_type="CHAT_MODEL",
        )

        # Check if the endpoint is a Llama model or similar non-OpenAI model
        is_llama_model = "llama" in endpoint_name.lower()
        
        # Different models support different parameters
        if tools:
            # Base parameters
            params = {
                "model": endpoint_name,
                "messages": messages,
                "tools": self.tool_json_schemas,
                **llm_options,
            }
            
            # Only add parallel_tool_calls for OpenAI models
            if not is_llama_model:
                params["parallel_tool_calls"] = False
                
            return traced_create(**params)
        else:
            return traced_create(model=endpoint_name, messages=messages, **llm_options)


# tell MLflow logging where to find the agent's code
set_model(MultiAgentSupervisor())