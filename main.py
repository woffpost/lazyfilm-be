from dotenv import load_dotenv
load_dotenv()

import os
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from anthropic import Anthropic

app = FastAPI(title="What to Watch Tonight API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

api_key = os.getenv("ANTHROPIC_API_KEY")
if api_key:
    api_key = api_key.strip().replace('"', '').replace("'", "")
anthropic_client = Anthropic(api_key=api_key)

# Описываем входные данные от фронтенда
class QuizAnswers(BaseModel):
    mood: str
    timing: str
    language: str
    custom_wish: Optional[str] = ""

# СТРОГАЯ СХЕМА ДЛЯ ИИ: Описываем, какой именно объект мы ждем от Клода
class MovieRecommendation(BaseModel):
    title: str = Field(description="Exact English title of the movie")
    year: int = Field(description="Release year of the movie")
    reason: str = Field(description="Compelling reason in Russian language why this movie fits the criteria")

class RecommendationList(BaseModel):
    movies: List[MovieRecommendation]

@app.post("/api/ai/recommend/")
async def get_ai_recommendations(answers: QuizAnswers):
    try:
        user_prompt = f"""
        User Quiz Results:
        - Current Mood/Context: {answers.mood}
        - Available Time: {answers.timing}
        - Preferred Language Environment: {answers.language}
        - Custom User Wishes: {answers.custom_wish}
        
        Select exactly 3 ideal movies. Write the 'reason' field strictly in Russian.
        """

        # МАГИЯ STRUCTURED OUTPUTS: Используем beta.messages.create c response_shape
        # Это заставляет Клода на уровне ядра выдавать идеальный, валидный JSON, соответствующий нашей Pydantic схеме
        response = anthropic_client.beta.messages.create(
            model="claude-3-5-sonnet-latest",
            max_tokens=1000,
            messages=[{"role": "user", "content": user_prompt}],
            response_shape={
                "type": "json_object",
                "schema": RecommendationList.model_json_schema()
            }
        )

        # Извлекаем уже готовый, распарсенный питоновский словарь! Больше никакого json.loads()!
        raw_content = response.content[0].text
        data = json.loads(raw_content)
        
        # Наш фронтенд ждет обычный массив [{title, year, reason}], поэтому достаем его из обертки
        return data.get("movies", [])

    except Exception as e:
        import traceback
        print("💥 КРИТИЧЕСКАЯ ОШИБКА НА БЭКЕНДЕ:")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
