from dotenv import load_dotenv
load_dotenv()

import os
import json
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from anthropic import Anthropic




app = FastAPI(title="What to Watch Tonight API")

# Настраиваем CORS, чтобы наш React-фронтенд мог спокойно делать запросы
app.add_middleware(
    CORSMiddleware,
    # Разрешаем оба порта, на которых может стартовать Vite
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:5174",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:5174",
        "https://cinebrowselite-be.onrender.com"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Инициализируем клиента Клода
api_key = os.getenv("ANTHROPIC_API_KEY")

if api_key:
    # .strip() удалит любые случайные невидимые пробелы, кавычки, \n или \r на концах строки
    api_key = api_key.strip().replace('"', '').replace("'", "")
else:
    print("❌ КРИТИЧЕСКАЯ ОШИБКА: API ключ Anthropic не найден в .env файле!")



anthropic_client = Anthropic(api_key=api_key)


# Описываем структуру анкеты, которую пришлет фронтенд
class QuizAnswers(BaseModel):
    mood: str          # Например: "heavy_day", "date_night", "scary"
    timing: str        # Например: "short", "standard", "epic"
    language: str      # Например: "ru", "en", "any"
    custom_wish: Optional[str] = "" # Любое текстовое пожелание пользователя


    # Системный промпт, который заставит Клода думать как кинокритик и отвечать строго в JSON
SYSTEM_PROMPT = """
You are an expert movie concierge and film critic. Your job is to recommend exactly 3 movies based on the user's emotional state, available time, language preference, and custom wishes.

You must look up real, existing movies and provide their correct TMDB (The Movie Database) IDs.
You MUST respond STRICTLY with a valid JSON array of objects. Do not include any conversational text, markdown formatting (like ```json), or explanations outside the JSON structure.

Expected JSON output format:
[
  {
    "id": 550, 
    "reason": "Short, compelling reason in Russian language explaining why this movie perfectly fits their current mood and criteria."
  }
]
"""

@app.post("/api/ai/recommend")
async def get_ai_recommendations(answers: QuizAnswers):
    try:
        # Формируем человеческое описание для ИИ на основе пришедших данных
        user_prompt = f"""
        User Quiz Results:
        - Current Mood/Context: {answers.mood}
        - Available Time: {answers.timing}
        - Preferred Language Environment: {answers.language}
        - Custom User Wishes: {answers.custom_wish}
        
        Please select 3 ideal movies for tonight. Write the 'reason' field strictly in Russian.
        """

        # Делаем официальный запрос к модели Claude 3.5 Sonnet
        response = anthropic_client.messages.create(
            model="claude-sonnet-4-6", # Используем актуальную Sonnet
            max_tokens=1000,
            system=SYSTEM_PROMPT,
            messages=[
                {"role": "user", "content": user_prompt}
            ]
        )

        # Извлекаем текстовый ответ Клода
        raw_text = response.content[0].text.strip()
        
        # На всякий случай очищаем от возможных markdown-оберток, если ИИ ослушался
        if raw_text.startswith("```json"):
            raw_text = raw_text.replace("```json", "").replace("```", "").strip()

        # Парсим строку в настоящий питоновский массив/словарь
        recommendations = json.loads(raw_text)
        return recommendations

    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="AI returned invalid JSON structure")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")
