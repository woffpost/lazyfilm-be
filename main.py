from dotenv import load_dotenv
load_dotenv()

import os
import json
import traceback  # ИСПОЛЬЗУЕМ ДЛЯ ВЫВОДА ПОЛНОГО ТРЕЙСБЭКА ОШИБКИ
from typing import List, Optional
import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
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

TMDB_TOKEN = os.getenv("TMDB_TOKEN") or os.getenv("VITE_TMDB_TOKEN")

class QuizAnswers(BaseModel):
    mood: str
    timing: str
    language: str
    custom_wish: Optional[str] = ""

SYSTEM_PROMPT = """
You are an expert movie concierge and film critic. Your job is to recommend exactly 3 movies based on the user's criteria.
You MUST respond STRICTLY with a valid JSON array of objects. Do not include markdown formatting or extra text.

Expected JSON output format:
[
  {
    "title": "Interstellar",
    "year": 2014,
    "reason": "Короткое описание на русском языке."
  }
]
"""

async def find_real_tmdb_id(title: str, year: int) -> Optional[int]:
    if not TMDB_TOKEN:
        print("⚠️ TMDB_TOKEN отсутствует в переменных окружения!")
        return None
        
    async with httpx.AsyncClient() as client:
        try:
            clean_token = TMDB_TOKEN.replace("Bearer ", "").strip()
            response = await client.get(
                "https://themoviedb.org",
                params={"query": title, "year": year},
                headers={"Authorization": f"Bearer {clean_token}"}
            )
            if response.status_code == 200:
                results = response.json().get("results", [])
                if results and len(results) > 0:
                    return results[0]["id"]  # Безопасно берем ID из первого элемента
            else:
                print(f"⚠️ TMDB API вернул статус {response.status_code}: {response.text}")
            return None
        except Exception as e:
            print(f"❌ Исключение в find_real_tmdb_id: {e}")
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

        # 1. СТРОГИЙ СИНТАКСИС SDK ИЗВЛЕЧЕНИЯ ТЕКСТА КЛОДА
        raw_text = response.content[0].text.strip()
        print(f"ℹ️ Получен сырой текст от Клода: {raw_text}")

        if raw_text.startswith("```json"):
            raw_text = raw_text.replace("```json", "").replace("```", "").strip()
        elif raw_text.startswith("```"):
            raw_text = raw_text.strip("`").strip()

        # 2. ПАРСИНГ JSON
        try:
            ai_output = json.loads(raw_text)
        except json.JSONDecodeError as je:
            print(f"❌ Ошибка парсинга JSON от Клода: {je}. Текст: {raw_text}")
            raise HTTPException(status_code=500, detail="AI returned malformed JSON content")

        # 3. ВЕРИФИКАЦИЯ ФИЛЬМОВ ЧЕРЕЗ TMDB
        verified_recommendations = []
        for item in ai_output:
            print(f"🔎 Ищем фильм: {item.get('title')} ({item.get('year')})")
            real_id = await find_real_tmdb_id(item.get("title"), item.get("year"))
            if real_id:
                verified_recommendations.append({
                    "id": real_id,
                    "reason": item.get("reason", "")
                })
        
        print(f"✅ Успешно верифицировано фильмов: {len(verified_recommendations)}")
        return verified_recommendations

    except Exception as e:
        # ЭТОТ БЛОК НАПЕЧАТАЕТ ПОЛНУЮ ОШИБКУ В КОНСОЛЬ RENDER
        print("💥 КРИТИЧЕСКАЯ ОШИБКА В ЭНДПОИНТЕ:")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
