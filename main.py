from dotenv import load_dotenv
load_dotenv()

import os
import json
import traceback
from typing import List, Optional
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

class QuizAnswers(BaseModel):
    mood: str
    timing: str
    language: str
    custom_wish: Optional[str] = ""

# УСИЛЕННЫЙ ПРОМПТ С ОБУЧЕНИЕМ НА ПРИМЕРАХ (FEW-SHOT PROMPTING)
SYSTEM_PROMPT = """
You are an expert movie concierge and film critic. Your job is to recommend exactly 3 movies based on the user's emotional state, timing, language, and custom wishes.

CRITICAL REQUIREMENT: You MUST provide the real, correct TMDB (The Movie Database) IDs for the movies. Do not hallucinate or guess digits.
Here are reference examples of real TMDB IDs you must use if relevant:
- Interstellar: 157336
- Inception: 27205
- Dune (2021): 438631
- Blade Runner 2049: 335984
- Malena (Monica Bellucci): 10515
- Irreversible (Monica Bellucci): 979
- The Matrix Reloaded: 604
- Spectre: 37724
- Shutter Island: 11324
- The Truman Show: 10447

You MUST respond STRICTLY with a valid JSON array of objects. Do not include markdown formatting like ```json, headers, or any conversational text.

Expected JSON output format:
[
  {
    "id": 157336, 
    "reason": "Короткое, убедительное описание на русском языке, почему этот фильм подходит."
  }
]
"""

@app.post("/api/ai/recommend/")
async def get_ai_recommendations(answers: QuizAnswers):
    try:
        user_prompt = f"""
        User Quiz Results:
        - Current Mood/Context: {answers.mood}
        - Available Time: {answers.timing}
        - Preferred Language Environment: {answers.language}
        - Custom User Wishes: {answers.custom_wish}
        
        Select exactly 3 ideal movies. Write the 'reason' field strictly in Russian. Ensure the TMDB IDs are 100% correct.
        """

        response = anthropic_client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1000,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_prompt}]
        )

        # Универсальное и безопасное извлечение текста
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

        # Очищаем от возможных markdown тегов ИИ
        if raw_text.startswith("```json"):
            raw_text = raw_text.replace("```json", "").replace("```", "").strip()
        elif raw_text.startswith("```"):
            raw_text = raw_text.strip("`").strip()

        try:
            verified_recommendations = json.loads(raw_text)
            
            # Простая валидация структуры
            if not isinstance(verified_recommendations, list):
                raise ValueError("Output is not a list")
                
            return verified_recommendations
            
        except json.JSONDecodeError as je:
            print(f"❌ Ошибка JSON: {je}. Текст: {raw_text}")
            raise HTTPException(status_code=500, detail="AI returned malformed JSON content")

    except Exception as e:
        print("💥 КРИТИЧЕСКАЯ ОШИБКА В ЭНДПОИНТЕ:")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
