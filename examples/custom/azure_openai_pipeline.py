from typing import List, Union, Generator, Iterator
from pydantic import BaseModel
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
import os
import json


class Pipeline:
    class Valves(BaseModel):
        # You can add your custom valves here.
        AZURE_OPENAI_API_KEY: str
        AZURE_OPENAI_ENDPOINT: str
        AZURE_OPENAI_DEPLOYMENT_NAME: str
        AZURE_OPENAI_API_VERSION: str

        class Config:
            arbitrary_types_allowed = True

    def __init__(self):
        try:
            self.name = "Azure OpenAI Pipeline"
            self.valves = self.Valves(
                **{
                    "AZURE_OPENAI_API_KEY": os.getenv("AZURE_OPENAI_API_KEY", "your-azure-openai-api-key-here"),
                    "AZURE_OPENAI_ENDPOINT": os.getenv("AZURE_OPENAI_ENDPOINT", "your-azure-openai-endpoint-here"),
                    "AZURE_OPENAI_DEPLOYMENT_NAME": os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "your-deployment-name-here"),
                    "AZURE_OPENAI_API_VERSION": os.getenv("AZURE_OPENAI_API_VERSION", "2024-08-01-preview")
                }
            )
        except Exception as e:
            print(f"Error initializing Azure OpenAI Pipeline: {str(e)}")
            import traceback
            print(f"Traceback: {traceback.format_exc()}")
            # 기본값으로 초기화
            self.name = "Azure OpenAI Pipeline"
            self.valves = self.Valves(
                **{
                    "AZURE_OPENAI_API_KEY": os.getenv("AZURE_OPENAI_API_KEY", "your-azure-openai-api-key-here"),
                    "AZURE_OPENAI_ENDPOINT": os.getenv("AZURE_OPENAI_ENDPOINT", "your-azure-openai-endpoint-here"),
                    "AZURE_OPENAI_DEPLOYMENT_NAME": os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "your-deployment-name-here"),
                    "AZURE_OPENAI_API_VERSION": os.getenv("AZURE_OPENAI_API_VERSION", "2024-08-01-preview")
                }
            )

    async def on_startup(self):
        print(f"on_startup:{__name__}")
        pass

    async def on_shutdown(self):
        print(f"on_shutdown:{__name__}")
        pass

    async def pipe(
            self, user_message: str, model_id: str, messages: List[dict], body: dict
    ) -> Union[str, Generator, Iterator]:
        print(f"pipe:{__name__}")
        print(f"Input messages: {json.dumps(messages, indent=2)}")
        print(f"User message: {user_message}")
        print(f"Body: {json.dumps(body, indent=2)}")

        try:
            client = AzureChatOpenAI(
                openai_api_key=self.valves.AZURE_OPENAI_API_KEY,
                openai_api_version=self.valves.AZURE_OPENAI_API_VERSION,
                azure_endpoint=self.valves.AZURE_OPENAI_ENDPOINT,
                azure_deployment=self.valves.AZURE_OPENAI_DEPLOYMENT_NAME,
                streaming=body.get("stream", False),
                temperature=body.get("temperature", 0.7),
                max_tokens=body.get("max_tokens", None)
            )

            allowed_params = {'messages', 'temperature', 'role', 'content', 'contentPart', 'contentPartImage',
                            'enhancements', 'data_sources', 'n', 'stream', 'stop', 'max_tokens', 'presence_penalty',
                            'frequency_penalty', 'logit_bias', 'user', 'function_call', 'functions', 'tools',
                            'tool_choice', 'top_p', 'log_probs', 'top_logprobs', 'response_format', 'seed'}

            if "user" in body and not isinstance(body["user"], str):
                body["user"] = body["user"]["id"] if "id" in body["user"] else str(body["user"])
            
            filtered_body = {k: v for k, v in body.items() if k in allowed_params}
            if len(body) != len(filtered_body):
                print(f"Dropped params: {', '.join(set(body.keys()) - set(filtered_body.keys()))}")

            raw_messages = filtered_body.get("messages", [])
            print(f"Raw messages: {json.dumps(raw_messages, indent=2)}")
            
            langchain_messages = []
            for msg in raw_messages:
                if msg["role"] == "system":
                    langchain_messages.append(SystemMessage(content=msg["content"]))
                elif msg["role"] == "user":
                    langchain_messages.append(HumanMessage(content=msg["content"]))
                elif msg["role"] == "assistant":
                    langchain_messages.append(AIMessage(content=msg["content"]))
            
            print(f"Langchain messages: {langchain_messages}")
            
            if body.get("stream", False):
                async def async_stream_generator():
                    try:
                        async for chunk in await client.astream(langchain_messages):
                            if hasattr(chunk, 'content'):
                                content = chunk.content
                                if content.startswith('data:image/'):
                                    print(f"Found image data in stream, length: {len(content)}")
                                
                                response_chunk = {
                                    "choices": [{
                                        "delta": {
                                            "role": "assistant",
                                            "content": content
                                        },
                                        "index": 0,
                                        "finish_reason": None
                                    }],
                                    "created": None,
                                    "model": self.valves.AZURE_OPENAI_DEPLOYMENT_NAME,
                                    "object": "chat.completion.chunk"
                                }
                                
                                try:
                                    json_str = json.dumps(response_chunk)
                                    print(f"Chunk size (bytes): {len(json_str)}")
                                    yield f"data: {json_str}\n\n"
                                except Exception as e:
                                    print(f"Error in JSON serialization: {str(e)}")
                                    if len(content) > 1024:
                                        print("Large content detected, splitting into chunks")
                                        chunk_size = 1024
                                        for i in range(0, len(content), chunk_size):
                                            sub_chunk = content[i:i + chunk_size]
                                            sub_response = {
                                                "choices": [{
                                                    "delta": {
                                                        "role": "assistant",
                                                        "content": sub_chunk
                                                    },
                                                    "index": 0,
                                                    "finish_reason": None
                                                }],
                                                "created": None,
                                                "model": self.valves.AZURE_OPENAI_DEPLOYMENT_NAME,
                                                "object": "chat.completion.chunk"
                                            }
                                            yield f"data: {json.dumps(sub_response)}\n\n"
                        
                        yield "data: [DONE]\n\n"
                    except Exception as e:
                        print(f"Error in stream generation: {str(e)}")
                        import traceback
                        print(f"Traceback: {traceback.format_exc()}")
                        yield f"data: {json.dumps({'error': str(e)})}\n\n"
                        yield "data: [DONE]\n\n"

                return async_stream_generator()
            else:
                response = await client.ainvoke(langchain_messages)
                print(f"Langchain response: {response}")
                
                result = {
                    "choices": [{
                        "message": {
                            "role": "assistant",
                            "content": response.content
                        },
                        "index": 0,
                        "finish_reason": "stop"
                    }],
                    "created": None,
                    "model": self.valves.AZURE_OPENAI_DEPLOYMENT_NAME,
                    "object": "chat.completion",
                    "usage": {"completion_tokens": 0, "prompt_tokens": 0, "total_tokens": 0}
                }
                print(f"Final response: {json.dumps(result, indent=2)}")
                return result
                
        except Exception as e:
            print(f"Error occurred: {str(e)}")
            import traceback
            print(f"Traceback: {traceback.format_exc()}")
            return f"Error: {str(e)}\n{traceback.format_exc()}" 