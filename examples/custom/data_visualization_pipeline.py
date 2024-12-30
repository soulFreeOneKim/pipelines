# -*- coding: utf-8 -*-
from typing import List, Union, Generator, Iterator
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import base64
import logging
from io import BytesIO

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('pipelines')

class Pipeline:
    def __init__(self):
        self.name = "Simple Plot Pipeline"
        logger.info(f"Initializing {self.name}")

    async def on_startup(self):
        logger.info(f"on_startup: {__name__}")

    async def on_shutdown(self):
        logger.info(f"on_shutdown: {__name__}")

    def create_plot(self, plot_type: str, title: str = 'Simple Plot') -> str:
        try:
            # Basic data
            x = [1, 2, 3, 4, 5]
            y = [2, 4, 6, 8, 10]
            
            # Create plot
            plt.figure()
            
            if plot_type == "scatter":
                plt.scatter(x, y)
            elif plot_type == "line":
                plt.plot(x, y)
            elif plot_type == "bar":
                plt.bar(x, y)
                
            plt.title(title)
            
            # Save to buffer
            buffer = BytesIO()
            plt.savefig(buffer, format='png')
            plt.close()
            buffer.seek(0)
            
            # Convert to base64
            image_base64 = base64.b64encode(buffer.getvalue()).decode()
            return f"data:image/png;base64,{image_base64}"
            
        except Exception as e:
            logger.error(f"Error creating plot: {str(e)}")
            plt.close()
            raise

    def pipe(self, user_message: str, model_id: str, messages: List[dict], body: dict) -> Union[str, Generator, Iterator]:
        try:
            command = user_message.strip().lower()
            
            # Get plot type
            plot_type = command.split()[0]
            if plot_type not in ["line", "scatter", "bar"]:
                yield "Please use: line, scatter, or bar"
                return
                
            # Get title if provided
            title = 'Simple Plot'
            if 'title=' in command:
                title = command.split('title=')[1].strip("'\"")
                
            # Create and return plot
            image_data = self.create_plot(plot_type, title)
            
            # The chart data for AI to analyze
            chart_data = {
                'type': plot_type,
                'title': title,
                'x_data': [1, 2, 3, 4, 5],
                'y_data': [2, 4, 6, 8, 10]
            }
            
            # Create AI analysis prompt
            analysis_prompt = f"""
분석할 차트 정보:
- 차트 유형: {chart_data['type']}
- 제목: {chart_data['title']}
- X축 데이터: {chart_data['x_data']}
- Y축 데이터: {chart_data['y_data']}

다음 형식으로 차트를 분석해주세요:
1. 차트의 전반적인 특징과 패턴 설명
2. 데이터의 의미와 관계성 분석
3. 주목할만한 포인트나 특이사항
"""
            # AI response would be generated here
            # For now, we'll create a placeholder for the result format
            result = f"### {title}\n\n"  # Title
            result += f"![{plot_type} plot]({image_data})\n\n"  # Chart
            result += f"**차트 분석**:\n"  # AI analysis would go here
            result += f"1. 차트 특징: [AI 분석 결과]\n"
            result += f"2. 데이터 분석: [AI 분석 결과]\n"
            result += f"3. 주요 포인트: [AI 분석 결과]\n\n"
            result += f"**데이터 정보**:\n"
            result += f"- X축 데이터: {chart_data['x_data']}\n"
            result += f"- Y축 데이터: {chart_data['y_data']}\n"
            
            yield result
            
        except Exception as e:
            logger.error(f"Error: {str(e)}")
            yield f"Error creating plot. Please try: line title='My Plot'"