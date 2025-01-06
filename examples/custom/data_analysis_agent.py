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
                        chat_history = self.get_session_history(session_id)
                        
                        # 시작 메시지
                        start_message = {
                            'choices': [{
                                'delta': {
                                    'role': 'assistant',
                                    'content': '🤔 분석을 시작합니다...\n\n'
                                },
                                'index': 0
                            }]
                        }
                        yield f"data: {json.dumps(start_message)}\n\n"
                        
                        for step in self.agent_with_chat_history.stream(
                            {"input": last_user_message, "chat_history": chat_history},
                            config={"configurable": {"session_id": session_id}},
                        ):
                            if isinstance(step, dict):
                                # 중간 단계 처리
                                if "intermediate_steps" in step:
                                    for intermediate_step in step["intermediate_steps"]:
                                        # 도구 실행 시작
                                        tool_message = {
                                            'choices': [{
                                                'delta': {
                                                    'role': 'assistant',
                                                    'content': '🔧 도구를 실행합니다...\n'
                                                },
                                                'index': 0
                                            }]
                                        }
                                        yield f"data: {json.dumps(tool_message)}\n\n"
                                        
                                        action, observation = self._process_intermediate_step(intermediate_step)
                                        
                                        if action and isinstance(action, dict):
                                            # 코드 실행 메시지
                                            if action.get("tool") == "python_repl_ast":
                                                code = action.get("tool_input", {}).get("query", "")
                                                code_message = {
                                                    'choices': [{
                                                        'delta': {
                                                            'role': 'assistant',
                                                            'content': f'```python\n{code}\n```\n'
                                                        },
                                                        'index': 0
                                                    }]
                                                }
                                                yield f"data: {json.dumps(code_message)}\n\n"
                                                
                                                # 실행 결과 처리
                                                result = self._execute_code(code, is_visualization=self._is_visualization_code(code))
                                                result_message = {
                                                    'choices': [{
                                                        'delta': {
                                                            'role': 'assistant',
                                                            'content': f'실행 결과:\n{result}\n\n'
                                                        },
                                                        'index': 0
                                                    }]
                                                }
                                                yield f"data: {json.dumps(result_message)}\n\n"
                                
                                # 최종 응답 처리
                                if "output" in step:
                                    output_content = f'📊 분석 결과:\n{step["output"]}\n'
                                    output_message = {
                                        'choices': [{
                                            'delta': {
                                                'role': 'assistant',
                                                'content': output_content
                                            },
                                            'index': 0
                                        }]
                                    }
                                    yield f"data: {json.dumps(output_message)}\n\n"
                        
                        # 종료 메시지
                        complete_message = {
                            'choices': [{
                                'delta': {
                                    'role': 'assistant',
                                    'content': '\n✅ 분석이 완료되었습니다.'
                                },
                                'index': 0
                            }]
                        }
                        yield f"data: {json.dumps(complete_message)}\n\n"
                        yield "data: [DONE]\n\n"
                                
                    except Exception as e:
                        error_msg = f"❌ 오류가 발생했습니다: {str(e)}"
                        yield f"data: {json.dumps({'choices': [{'delta': {'role': 'assistant', 'content': error_msg}, 'index': 0}]})}\n\n"
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

    def _process_intermediate_step(self, intermediate_step):
        """중간 단계 처리를 위한 헬퍼 메서드"""
        action = None
        observation = None
        
        if hasattr(intermediate_step, "tool"):
            action = {
                "tool": intermediate_step.tool,
                "tool_input": intermediate_step.tool_input
            }
            observation = intermediate_step.observation
        elif isinstance(intermediate_step, tuple) and len(intermediate_step) == 2:
            action, observation = intermediate_step
            if hasattr(action, "tool"):
                action = {
                    "tool": action.tool,
                    "tool_input": action.tool_input
                }
        
        return action, observation

    def _is_visualization_code(self, code: str) -> bool:
        """시각화 코드 여부 확인"""
        viz_keywords = [
            "plt.show()",
            "sns.barplot",
            "plt.figure",
            "sns.set",
            "matplotlib",
            "seaborn"
        ]
        return any(keyword in code for keyword in viz_keywords)

    def _execute_code(self, code: str, is_visualization: bool = False):
        """코드 실행 및 결과 반환"""
        try:
            if is_visualization:
                return self._execute_visualization_code(code)
            else:
                return self._execute_regular_code(code)
        except Exception as e:
            return f"실행 오류: {str(e)}" 

    def _execute_visualization_code(self, code: str) -> str:
        """시각화 코드 실행 및 이미지 생성"""
        try:
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
            
            # 정리
            buffer.close()
            plt.close('all')
            
            return f"![plot](data:image/png;base64,{image_base64})"
            
        except Exception as e:
            return f"시각화 오류: {str(e)}"

    def _execute_regular_code(self, code: str) -> str:
        """일반 Python 코드 실행"""
        try:
            # 로컬 네임스페이스에서 코드 실행
            local_vars = {'df': self.df}
            local_vars.update(globals())
            local_vars.update(locals())
            
            # StringIO를 사용하여 출력 캡처
            from io import StringIO
            import sys
            
            output_buffer = StringIO()
            stdout_backup = sys.stdout
            sys.stdout = output_buffer
            
            try:
                exec(code, local_vars, local_vars)
                output = output_buffer.getvalue()
            finally:
                sys.stdout = stdout_backup
                output_buffer.close()
            
            # 실행 결과가 있는 경우 반환
            if output.strip():
                return output.strip()
            
            # 실행 결과가 없는 경우, 마지막 실행된 표현식의 결과 반환
            last_expression = code.strip().split('\n')[-1]
            try:
                result = eval(last_expression, local_vars, local_vars)
                return str(result)
            except:
                return "코드가 성공적으로 실행되었습니다."
                
        except Exception as e:
            return f"실행 오류: {str(e)}" 