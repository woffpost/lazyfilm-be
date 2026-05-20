from dotenv import load_dotenv
load_dotenv()

import os
import json
from typing import List, Optional
import httpx 
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from anthropic import Anthropic


app = FastAPI(title="What to Watch Tonight API")

# Настраиваем CORS, чтобы наш React-фронтенд мог спокойно делать запросы
app.add_middleware(
    CORSMiddleware,
    # Разрешаем оба порта, на которых может стартовать Vite
    allow_origins=["*"],  # РАЗРЕШАЕМ ВСЁ! Любые порты (5173, 5174, 5175) и любой Vercel-домен
    allow_credentials=False,  # Обязательно False, если origins=["*"]
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

TMDB_TOKEN = os.getenv("VITE_TMDB_TOKEN") or os.getenv("TMDB_TOKEN")


# Описываем структуру анкеты, которую пришлет фронтенд
class QuizAnswers(BaseModel):
    mood: str          # Например: "heavy_day", "date_night", "scary"
    timing: str        # Например: "short", "standard", "epic"
    language: str      # Например: "ru", "en", "any"
    custom_wish: Optional[str] = "" # Любое текстовое пожелание пользователя


    # Системный промпт, который заставит Клода думать как кинокритик и отвечать строго в JSON
SYSTEM_PROMPT = """
You are an expert movie concierge and film critic. Your job is to recommend exactly 3 movies based on the user's criteria.

You MUST respond STRICTLY with a valid JSON array of objects. Do not include markdown formatting or extra text.
For each movie, provide the exact English title and the release year so we can programmatically look up their IDs.

Expected JSON output format:
[
  {
    "id": 550, 
    "reason": "Short, compelling reason in Russian language explaining why this movie perfectly fits their current mood and criteria."
  }
]
"""

# Вспомогательная функция для поиска реального ID в TMDB по названию и году
async def find_real_tmdb_id(title: str, year: int) -> Optional[int]:
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(
                "https://themoviedb.org",
                params={"query": title, "year": year},
                headers={"Authorization": f"Bearer {TMDB_TOKEN.replace('Bearer ', '') if TMDB_TOKEN else ''}"}
            )
            if response.status_code == 200:
                results = response.json().get("results", [])
                if results:
                    return results[0]["id"] # Берем ID самого первого, точного совпадения
            return None
        except Exception as e:
            print(f"Ошибка поиска TMDB: {e}")
            return None
        
@app.post("/api/ai/recommend/")
async def get_ai_recommendations(answers: QuizAnswers):
    try:
        user_prompt = f"""
        User Quiz Results:
        - Current Mood/Context: {answers.mood}
        - Available Time: {answers.timing}
        - Preferred Language Environment: {answers.language}
        - Custom User Wishes: {answers.custom_wish}
        
        Select 3 ideal movies. Write the 'reason' field strictly in Russian.
        """

        response = anthropic_client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1000,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_prompt}]
        )

        raw_text = response.content[0].text.strip() if isinstance(response.content, list) else response.content.text.strip()
        if raw_text.startswith("```json"):
            raw_text = raw_text.replace("```json", "").replace("```", "").strip()

        ai_output = json.loads(raw_text)
        
        # МАГИЯ ВЕРИФИКАЦИИ: перебираем то, что вернул Клод, и ищем железные ID
        verified_recommendations = []
        for item in ai_output:
            real_id = await find_real_tmdb_id(item["title"], item["year"])
            if real_id:
                verified_recommendations.append({
                    "id": real_id,
                    "reason": item["reason"]
                })
        
        return verified_recommendations

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")
