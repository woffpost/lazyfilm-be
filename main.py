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
LANGUAGE_NAMES = {"ru": "Russian", "en": "English", "ro": "Romanian"}

class QuizAnswers(BaseModel):
    mood: str
    timing: str
    language: str
    custom_wish: Optional[str] = ""
    ui_language: Optional[str] = "en"

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
        reason_language = LANGUAGE_NAMES.get(answers.ui_language or "en", "English")
        user_prompt = f"""
        User Quiz Results:
        - Current Mood/Context: {answers.mood}
        - Available Time: {answers.timing}
        - Preferred Language Environment: {answers.language}
        - Custom User Wishes: {answers.custom_wish}

        Select exactly 3 ideal movies. Write the 'reason' field strictly in {reason_language}.
        """

        # Structured outputs через tool_use — единственный правильный способ в Anthropic SDK
        response = anthropic_client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1000,
            tools=[{
                "name": "recommend_movies",
                "description": "Return movie recommendations based on user preferences",
                "input_schema": RecommendationList.model_json_schema()
            }],
            tool_choice={"type": "tool", "name": "recommend_movies"},
            messages=[{"role": "user", "content": user_prompt}]
        )

        tool_block = next(b for b in response.content if b.type == "tool_use")
        data = tool_block.input
        return data.get("movies", [])

    except Exception as e:
        import traceback
        print("💥 КРИТИЧЕСКАЯ ОШИБКА НА БЭКЕНДЕ:")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
