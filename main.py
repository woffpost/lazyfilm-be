from dotenv import load_dotenv
load_dotenv()

import os
import json
import traceback
from typing import List, Optional
import requests  # Железобетонная кодировка URL на любых Linux-серверах
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from anthropic import Anthropic

app = FastAPI(title="What to Watch Tonight API")

# Сквозные CORS-настройки для любых локальных портов и Vercel-доменов
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

# Поддерживаем любые варианты названий переменных на Render
TMDB_TOKEN = os.getenv("TMDB_TOKEN") or os.getenv("VITE_TMDB_TOKEN") or os.getenv("tmdb_token")

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

def find_real_tmdb_id(title: str, year: int) -> Optional[int]:
    if not TMDB_TOKEN:
        print("❌ ОШИБКА: TMDB_TOKEN не найден в переменных окружения Render!")
        return None
        
    try:
        clean_token = TMDB_TOKEN.replace("Bearer ", "").strip()
        
        # Очищаем название от кавычек-ёлочек ИИ для точности поиска в TMDB
        clean_title = title.replace('"', '').replace("'", "").replace("«", "").replace("»", "").strip()
        
        response = requests.get(
            "https://themoviedb.org",
            params={"query": clean_title, "year": year},
            headers={"Authorization": f"Bearer {clean_token}"},
            timeout=5
        )
        
        if response.status_code == 200:
            results = response.json().get("results", [])
            # ИСПРАВЛЕНО: Безопасно забираем ID из ПЕРВОГО элемента массива результатов
            if results and len(results) > 0:
                return results[0]["id"] 
        else:
            print(f"⚠️ TMDB API вернул статус {response.status_code}: {response.text}")
        return None
    except Exception as e:
        print(f"❌ Исключение в find_real_tmdb_id: {e}")
        return None

@app.post("/api/ai/recommend/")
async def get_ai_recommendations(answers: QuizAnswers):
    try:
        if not TMDB_TOKEN:
            raise HTTPException(status_code=500, detail="Server Configuration Error: TMDB_TOKEN is missing on Render.")

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

        # ИСПРАВЛЕНО: Универсальное извлечение текста, неуязвимое к версиям Anthropic SDK
        content_obj = response.content
        raw_text = ""
        
        if isinstance(content_obj, list):
            if len(content_obj) > 0 and hasattr(content_obj[0], 'text'):
                raw_text = content_obj[0].text.strip()
            else:
                raw_text = str(content_obj).strip()
        elif hasattr(content_obj, 'text'):
            raw_text = content_obj.text.strip()
        else:
            raw_text = str(content_obj).strip()

        print(f"ℹ️ Успешно извлечен сырой текст: {raw_text}")

        # Счищаем возможную markdown-обертку Клода
        if raw_text.startswith("```json"):
            raw_text = raw_text.replace("```json", "").replace("```", "").strip()
        elif raw_text.startswith("```"):
            raw_text = raw_text.strip("`").strip()

        try:
            ai_output = json.loads(raw_text)
        except json.JSONDecodeError as je:
            print(f"❌ Ошибка JSON: {je}")
            raise HTTPException(status_code=500, detail=f"AI malformed JSON content: {raw_text}")

        # Верификация через TMDB по названиям строк
        verified_recommendations = []
        for item in ai_output:
            print(f"🔎 Поиск фильма: {item.get('title')} ({item.get('year')})")
            real_id = find_real_tmdb_id(item.get("title"), item.get("year"))
            if real_id:
                verified_recommendations.append({
                    "id": real_id,
                    "reason": item.get("reason", "")
                })
        
        if len(verified_recommendations) == 0:
            raise HTTPException(
                status_code=500, 
                detail=f"TMDB Verification Failed. 0 valid IDs found. Raw text: {raw_text}"
            )
        
        return verified_recommendations

    except HTTPException as he:
        raise he
    except Exception as e:
        print("💥 КРИТИЧЕСКАЯ ОШИБКА В ЭНДПОИНТЕ:")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
