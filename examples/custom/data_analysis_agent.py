from typing import List, Union, Generator, Iterator
from pydantic import BaseModel
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
import os
import json
import pandas as pd
from langchain_experimental.tools import PythonAstREPLTool
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain.agents.agent_types import AgentType
from langchain_openai import ChatOpenAI
from utilities.message import AgentStreamParser, AgentCallbacks
import matplotlib
import matplotlib.pyplot as plt
import io
import base64
from io import BytesIO
from langchain.agents import AgentExecutor
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder



class Pipeline:
    class Valves(BaseModel):
        # You can add your custom valves here.
        AZURE_OPENAI_API_KEY: str
        AZURE_OPENAI_ENDPOINT: str
        AZURE_OPENAI_DEPLOYMENT_NAME: str
        AZURE_OPENAI_API_VERSION: str
        TARGET_DATA_SOURCE: str

        model_config = {
            "arbitrary_types_allowed": True,
            "json_schema_extra": {
                "examples": [
                    {
                        "AZURE_OPENAI_API_KEY": "your-azure-openai-api-key-here",
                        "AZURE_OPENAI_ENDPOINT": "your-azure-openai-endpoint-here",
                        "AZURE_OPENAI_DEPLOYMENT_NAME": "your-deployment-name-here",
                        "AZURE_OPENAI_API_VERSION": "2024-08-01-preview",
                        "TARGET_DATA_SOURCE": "path/to/data.csv"
                    }
                ]
            }
        }

    def __init__(self):
        try:
            # 현재 파일의 절대 경로를 기준으로 data 폴더 경로 설정
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(os.path.dirname(current_dir))
            csv_path = os.path.join(project_root, 'pipelines', 'data', 'titanic.csv')
            
            # 경로 정규화
            csv_path = os.path.normpath(csv_path)
            print(f"Loading data from: {csv_path}")
            
            self.name = "Data Analysis Agent"
            self.valves = self.Valves(
                **{
                    "AZURE_OPENAI_API_KEY": os.getenv("AZURE_OPENAI_API_KEY", "your-azure-openai-api-key-here"),
                    "AZURE_OPENAI_ENDPOINT": os.getenv("AZURE_OPENAI_ENDPOINT", "your-azure-openai-endpoint-here"),
                    "AZURE_OPENAI_DEPLOYMENT_NAME": os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "your-deployment-name-here"),
                    "AZURE_OPENAI_API_VERSION": os.getenv("AZURE_OPENAI_API_VERSION", "2024-08-01-preview"),
                    "TARGET_DATA_SOURCE": csv_path
                }
            )
            # DataFrame은 startup에서 초기화될 예정
            self.df = None
            self.python_tool = None
            self.llm = None
            self.client = None
            self.agent = None
            self.session_store = {}
            self.agent_with_chat_history = None
        except Exception as e:
            print(f"Error in initialization: {str(e)}")
            import traceback
            print(f"Traceback: {traceback.format_exc()}")
            pass

    async def on_startup(self):
        print(f"on_startup:{__name__}")
        try:
            
            if os.path.exists(self.valves.TARGET_DATA_SOURCE):
                # CSV 파일을 DataFrame으로 읽기
                self.df = pd.read_csv(self.valves.TARGET_DATA_SOURCE)
                print(f"Successfully loaded DataFrame with shape: {self.df.shape}")
            else:
                print(f"Warning: {self.valves.TARGET_DATA_SOURCE} file not found")
                self.df = pd.DataFrame()  # 빈 DataFrame 생성

            self.python_tool = PythonAstREPLTool()
            self.python_tool.locals["df"] = self.df

            self.client = AzureChatOpenAI(
                openai_api_key=self.valves.AZURE_OPENAI_API_KEY,
                openai_api_version=self.valves.AZURE_OPENAI_API_VERSION,
                azure_endpoint=self.valves.AZURE_OPENAI_ENDPOINT,
                azure_deployment=self.valves.AZURE_OPENAI_DEPLOYMENT_NAME,
                temperature=0
            )

            self.llm = AzureChatOpenAI(
                openai_api_key=self.valves.AZURE_OPENAI_API_KEY,
                openai_api_version=self.valves.AZURE_OPENAI_API_VERSION,
                azure_endpoint=self.valves.AZURE_OPENAI_ENDPOINT,
                azure_deployment=self.valves.AZURE_OPENAI_DEPLOYMENT_NAME,
                temperature=0,
                streaming=True
            )

            # prompt = ChatPromptTemplate.from_messages([
            #     ("system", """You are a professional data analyst and expert in Pandas. 
            #     You must use Pandas DataFrame(`df`) to answer user's request.

            #     [IMPORTANT] DO NOT create or overwrite the `df` variable in your code.

            #     If you are willing to generate visualization code, please use `plt.show()` at the end of your code.
            #     I prefer seaborn code for visualization, but you can use matplotlib as well.

            #     <Visualization Preference>
            #     - `muted` cmap, white background, and no grid for your visualization.
            #     Recommend to set palette parameter for seaborn plot.

            #     Make sure to use the `pdf_search` tool for searching information from the PDF document.
            #     If you can't find the information from the PDF document, use the `search` tool for searching information from the web."""),
            #     MessagesPlaceholder(variable_name="chat_history", optional=True),
            #     ("human", "{input}"),
            #     MessagesPlaceholder(variable_name="agent_scratchpad")
            # ])
            
            
            self.agent = create_pandas_dataframe_agent(
                llm=self.llm,
                df=self.df,
                verbose=True,
                agent_type="tool-calling",
                max_iterations=5,
                return_intermediate_steps=True,
                allow_dangerous_code=True,
                prefix="You are a professional data analyst and expert in Pandas. "
                "You must use Pandas DataFrame(`df`) to answer user's request. "
                "\n\n[IMPORTANT] DO NOT create or overwrite the `df` variable in your code. \n\n"
                "If you are willing to generate visualization code, please use `plt.show()` at the end of your code. "
                "I prefer seaborn code for visualization, but you can use matplotlib as well."
                "\n\n<Visualization Preference>\n"
                "- `muted` cmap, white background, and no grid for your visualization."
                "\nRecomment to set palette parameter for seaborn plot.",
            )

            self.agent_with_chat_history = RunnableWithMessageHistory(
                self.agent,
                # 대화 session_id
                self.get_session_history,
                # 프롬프트의 질문이 입력되는 key: "input"
                input_messages_key="input",
                # 프롬프트의 메시지가 입력되는 key: "chat_history"
                history_messages_key="chat_history",
)
        except Exception as e:
            print(f"Error in agent initialization: {str(e)}")
            import traceback
            print(f"Traceback: {traceback.format_exc()}")

    async def on_shutdown(self):
        print(f"on_shutdown:{__name__}")
        # DataFrame 정리
        if hasattr(self, 'df'):
            del self.df

    def get_session_history(self, session_id):
        """세션 ID에 해당하는 대화 기록을 반환합니다. 없으면 새로 생성합니다."""
        if session_id not in self.session_store:
            self.session_store[session_id] = ChatMessageHistory()
        return self.session_store[session_id]

    def pipe(
            self, user_message: str, model_id: str, messages: List[dict], body: dict
    ) -> Union[str, Generator, Iterator]:
        print(f"### pipe:{__name__}")
        print(f"### model_id: {model_id}")
        print(f"### messages: {json.dumps(messages, indent=2)}")
        print(f"### User message: {user_message}")
        print(f"### Body: {json.dumps(body, indent=2)}")
        session_id = model_id + "_" + body.get("user",{}).get("id")
        print(f"### session_id: {session_id}")
        try:
            # DataFrame 상태 확인
            if self.df is None or self.agent is None:
                print("Warning: DataFrame or Agent is not initialized")
                self.on_startup()  # DataFrame과 Agent 초기화 시도

            if self.agent is None:
                raise Exception("Failed to initialize agent")

            # matplotlib 설정
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            import io
            import base64

            # 마지막 user 메시지를 가져옴
            last_user_message = None
            for msg in reversed(messages):
                if msg["role"] == "user":
                    last_user_message = msg["content"]
                    break
            
            if last_user_message is None:
                last_user_message = user_message

            # 이모지 타이틀이나 태그 생성 요청인지 확인
            is_emoji_request = "emoji" in user_message.lower() or "title" in user_message.lower()
            is_tag_request = "tags" in user_message.lower() or "categorizing" in user_message.lower()

            if is_emoji_request or is_tag_request:
                # 이모지/태그 요청은 client를 사용하여 직접 응답
                messages = [
                    SystemMessage(content="You are a helpful assistant that generates concise titles and tags."),
                    HumanMessage(content=last_user_message)
                ]
                response = self.client.invoke(messages)
                content = response.content

                # 마크다운 코드 블록 제거
                if content and content.startswith("```") and content.endswith("```"):
                    content = content.strip("`").strip()
                    if content.startswith("json"):
                        content = content[4:].strip()
                
                result = {
                    "choices": [{
                        "message": {
                            "role": "assistant",
                            "content": content
                        },
                        "index": 0,
                        "finish_reason": "stop"
                    }],
                    "created": None,
                    "model": self.valves.AZURE_OPENAI_DEPLOYMENT_NAME,
                    "object": "chat.completion",
                    "usage": {"completion_tokens": 0, "prompt_tokens": 0, "total_tokens": 0}
                }
                return result

            if body.get("stream", False):
                def stream_generator():
                    try:
                        import traceback  # traceback 모듈을 함수 내부로 이동
                        
                        chat_history = self.get_session_history(session_id)
                        for step in self.agent_with_chat_history.stream(
                            {"input": last_user_message,
                             "chat_history": chat_history},
                            config={"configurable": {"session_id": session_id}},
                        ):
                            print(f"### self.session_store: {chat_history.messages}")
                            print(f"Step type: {type(step)}")
                            if isinstance(step, dict):
                                try:
                                    print(f"Step contents: {json.dumps({k: str(v) for k, v in step.items()}, indent=2)}")
                                except:
                                    print("Could not serialize step contents")
                                
                                # Python 코드와 실행 결과 처리
                                if "intermediate_steps" in step:
                                    try:
                                        steps_info = []
                                        for s in step["intermediate_steps"]:
                                            if hasattr(s, "__dict__"):
                                                steps_info.append(str(s.__dict__))
                                            else:
                                                steps_info.append(str(s))
                                        print(f"Found intermediate_steps: {steps_info}")
                                    except Exception as e:
                                        print(f"Error processing intermediate steps: {str(e)}")

                                    for intermediate_step in step["intermediate_steps"]:
                                        # ToolAgentAction 객체 처리
                                        action = None
                                        observation = None
                                        
                                        if hasattr(intermediate_step, "tool"):
                                            # ToolAgentAction 객체인 경우
                                            action = {
                                                "tool": intermediate_step.tool,
                                                "tool_input": intermediate_step.tool_input
                                            }
                                            observation = intermediate_step.observation
                                        elif isinstance(intermediate_step, tuple) and len(intermediate_step) == 2:
                                            # 튜플인 경우
                                            action, observation = intermediate_step
                                            if hasattr(action, "tool"):
                                                action = {
                                                    "tool": action.tool,
                                                    "tool_input": action.tool_input
                                                }
                                        
                                        print(f"Action type: {type(action)}")
                                        print(f"Action content: {action}")
                                        
                                        if action and isinstance(action, dict) and action.get("tool") == "python_repl_ast":
                                            code = action.get("tool_input", {}).get("query", "")
                                            print(f"Found Python code to execute: {code}")
                                            
                                            if code:
                                                # 시각화 코드 감지를 위한 키워드 목록
                                                viz_keywords = [
                                                    "plt.show()",
                                                    "sns.barplot",
                                                    "plt.figure",
                                                    "sns.set",
                                                    "matplotlib",
                                                    "seaborn"
                                                ]
                                                
                                                # 시각화 코드 감지
                                                is_viz_code = any(keyword in code for keyword in viz_keywords)
                                                print(f"Is visualization code: {is_viz_code}")
                                                print(f"Detected keywords: {[kw for kw in viz_keywords if kw in code]}")
                                                
                                                if is_viz_code:
                                                    try:
                                                        print("Executing visualization code...")
                                                        
                                                        # matplotlib 설정
                                                        matplotlib.use('Agg')
                                                        plt.style.use('default')
                                                        plt.close('all')
                                                        
                                                        # 로컬 네임스페이스에서 코드 실행
                                                        local_vars = {'df': self.df}
                                                        local_vars.update(globals())
                                                        local_vars.update(locals())
                                                        exec(code, local_vars, local_vars)
                                                        
                                                        # 이미지를 버퍼에 저장
                                                        buffer = BytesIO()
                                                        plt.gcf().set_size_inches(8, 6)  # 그래프 크기를 8x6 인치로 설정
                                                        plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
                                                        buffer.seek(0)
                                                        image_base64 = base64.b64encode(buffer.getvalue()).decode()

                                                        image_base64 = f"data:image/png;base64,{image_base64}"


                                                        # 정리
                                                        buffer.close()
                                                        plt.close('all')
                                                        

                                                        yield f"![plot]({image_base64})\n\n"
                                                        
                                                    except Exception as viz_error:
                                                        import traceback
                                                        print(f"Visualization error: {str(viz_error)}")
                                                        print(f"Traceback: {traceback.format_exc()}")
                                                        error_response = {
                                                            "choices": [{
                                                                "delta": {
                                                                    "role": "assistant",
                                                                    "content": f"\nError generating visualization: {str(viz_error)}\n"
                                                                },
                                                                "index": 0,
                                                                "finish_reason": None
                                                            }]
                                                        }
                                                        yield f"data: {json.dumps(error_response)}\n\n"
                                                else:
                                                    # 일반 코드 실행
                                                    try:
                                                        # 로컬 네임스페이스에서 코드 실행
                                                        local_vars = {'df': self.df}
                                                        local_vars.update(globals())
                                                        local_vars.update(locals())
                                                        exec(code, local_vars, local_vars)
                                                        
                                                        # survival_rate_by_gender가 생성되었다면 전역 변수로 저장
                                                        if 'survival_rate_by_gender' in local_vars:
                                                            globals()['survival_rate_by_gender'] = local_vars['survival_rate_by_gender']
                                                        
                                                        yield f"data: {json.dumps({'choices': [{'delta': {'role': 'assistant', 'content': str(observation)}, 'index': 0}]})}\n\n"
                                                    except Exception as exec_error:
                                                        import traceback
                                                        print(f"Code execution error: {str(exec_error)}")
                                                        print(f"Traceback: {traceback.format_exc()}")
                                                        error_response = {
                                                            "choices": [{
                                                                "delta": {
                                                                    "role": "assistant",
                                                                    "content": f"\nError executing code: {str(exec_error)}\n"
                                                                },
                                                                "index": 0,
                                                                "finish_reason": None
                                                            }]
                                                        }
                                                        yield f"data: {json.dumps(error_response)}\n\n"
                                
                                # 최종 응답 처리
                                if "output" in step:
                                    output = str(step["output"]).strip()
                                    if output:
                                        # 마크다운 이미지 참조 제거
                                        output = output.replace("![", "").replace("](attachment://survival_rate_plot.png)", "")
                                        
                                        # 응답 전송
                                        yield f"data: {json.dumps({'choices': [{'delta': {'role': 'assistant', 'content': output}, 'index': 0}]})}\n\n"
                            
                            else:
                                # 기타 스텝 처리
                                yield f"data: {json.dumps({'choices': [{'delta': {'role': 'assistant', 'content': str(step)}, 'index': 0}]})}\n\n"
                        
                        # 스트림 종료
                        yield "data: [DONE]\n\n"
                                
                    except Exception as e:
                        import traceback
                        print(f"Error in stream_generator: {str(e)}")
                        print(f"Traceback: {traceback.format_exc()}")
                        yield f"data: {json.dumps({'choices': [{'delta': {'role': 'assistant', 'content': f'Error: {str(e)}'}, 'index': 0}]})}\n\n"
                        yield "data: [DONE]\n\n"
                
                return stream_generator()
            else:
                # 비스트리밍 모드
                chat_history = self.get_session_history(session_id)
                response = self.agent.invoke(
                    {"input": last_user_message,
                     "chat_history": chat_history},
                    config={"configurable": {"session_id": session_id}},
                )
                content = None
                if isinstance(response, dict):
                    if "output" in response:
                        content = str(response["output"])
                    elif "intermediate_steps" in response:
                        steps = response["intermediate_steps"]
                        contents = []
                        for step in steps:
                            if isinstance(step, tuple) and len(step) == 2:
                                action, observation = step
                                if isinstance(action, dict) and action.get("tool") == "python_repl_ast":
                                    tool_input = action.get("tool_input", {})
                                    contents.append(f"```python\n{tool_input.get('query', '')}\n```")
                                    if observation:
                                        contents.append(f"실행 결과:\n{observation}")
                        content = "\n".join(contents)
                else:
                    content = str(response)
                
                print(f"Agent response: {content}")
                
                result = {
                    "choices": [{
                        "message": {
                            "role": "assistant",
                            "content": content
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