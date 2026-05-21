from dotenv import load_dotenv
load_dotenv()

import os
import json
import re
import traceback
from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from anthropic import Anthropic

app = FastAPI(title="What to Watch Tonight API")

# Полноценный Wildcard CORS для продакшена и любых локальных Vite-портов
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

class QuizAnswers(BaseModel):
    mood: str
    timing: str
    language: str
    custom_wish: Optional[str] = ""

SYSTEM_PROMPT = """
You are an expert movie concierge and film critic. Your job is to recommend exactly 3 movies based on the user's criteria.

You MUST respond with a valid JSON array of objects. Do not include markdown formatting or extra text outside the array.
For each movie, provide the exact English title and release year so the frontend can look up their correct IDs.

Expected JSON output format:
[
  {
    "title": "Malena",
    "year": 2000,
    "reason": "Описание фильма на русском языке."
  }
]
"""

# ИСПРАВЛЕНО: Путь теперь строго совпадает с фронтендом (/api/ai/recommend/)
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

        response = anthropic_client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1000,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_prompt}]
        )

        # Безопасное извлечение текстового блока Клода
        content_obj = response.content
        raw_text = ""
        
        if isinstance(content_obj, list):
            if len(content_obj) > 0 and hasattr(content_obj, 'text'):
                raw_text = content_obj.text.strip()
            else:
                raw_text = str(content_obj).strip()
        elif hasattr(content_obj, 'text'):
            raw_text = content_obj.text.strip()
        else:
            raw_text = str(content_obj).strip()

        # Регулярным выражением вырезаем только массив данных
        match = re.search(r'\[.*\]', raw_text, re.DOTALL)
        if not match:
            raise HTTPException(status_code=500, detail="AI response did not contain a JSON array.")

        return json.loads(match.group(0).strip())

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
