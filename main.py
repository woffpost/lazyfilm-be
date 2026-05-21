from dotenv import load_dotenv
load_dotenv()

import os
import json
import re  # ИСПОЛЬЗУЕМ РЕГУЛЯРНЫЕ ВЫРАЖЕНИЯ ДЛЯ ВЫРЕЗАНИЯ JSON
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

You MUST respond with a valid JSON array of objects. Even if you include conversational text, ensure the JSON array is enclosed in [ and ].

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

        # Извлекаем текст Клода через официальный интерфейс SDK
        raw_text = ""
        if hasattr(response, 'content') and isinstance(response.content, list) and len(response.content) > 0:
            raw_text = response.content[0].text.strip()
        elif hasattr(response, 'content') and hasattr(response.content, 'text'):
            raw_text = response.content.text.strip()
        else:
            raw_text = str(response).strip()

        print(f"ℹ️ Успешно извлечен текст от Клода: {raw_text}")

        # ПРОМЫШЛЕННЫЙ ХАК: Ищем первый попавшийся массив [...] с помощью RegEx
        match = re.search(r'\[.*\]', raw_text, re.DOTALL)
        if not match:
            print(f"❌ Критическая ошибка: В ответе ИИ вообще не найден массив []. Текст: {raw_text}")
            raise HTTPException(status_code=500, detail="AI response did not contain a valid JSON array structure.")

        # Вырезаем только ту часть, которая находится внутри скобок
        clean_json_text = match.group(0).strip()

        try:
            verified_recommendations = json.loads(clean_json_text)
            
            if not isinstance(verified_recommendations, list):
                raise ValueError("Parsed content is not a JSON array list")
                
            return verified_recommendations
            
        except json.JSONDecodeError as je:
            print(f"❌ Ошибка JSON: {je}. Очищенный текст: {clean_json_text}")
            raise HTTPException(status_code=500, detail=f"AI returned malformed JSON content: {clean_json_text}")

    except HTTPException as he:
        raise he
    except Exception as e:
        print("💥 КРИТИЧЕСКАЯ ОШИБКА В ЭНДПОИНТЕ:")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
