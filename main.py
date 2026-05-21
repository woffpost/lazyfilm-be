from dotenv import load_dotenv
load_dotenv()

import os
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from anthropic import Anthropic

app = FastAPI(title="CineBrowse API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

api_key = os.getenv("ANTHROPIC_API_KEY", "").strip().replace('"', '').replace("'", "")
tmdb_token = os.getenv("TMDB_TOKEN", "").strip()

anthropic_client = Anthropic(api_key=api_key)

TMDB_BASE = "https://api.themoviedb.org/3"
LANGUAGE_NAMES = {"ru": "Russian", "en": "English", "ro": "Romanian"}
TMDB_LANG = {"ru": "ru-RU", "en": "en-US", "ro": "ro-RO"}


class QuizAnswers(BaseModel):
    mood: str
    timing: str
    language: str
    custom_wish: Optional[str] = ""
    ui_language: Optional[str] = "en"


class MovieRecommendation(BaseModel):
    title: str = Field(description="Exact English title of the movie")
    year: int = Field(description="Release year of the movie")
    reason: str = Field(description="Why this movie fits the user's criteria")


class RecommendationList(BaseModel):
    movies: List[MovieRecommendation]


def enrich_with_tmdb(rec: dict, tmdb_lang: str) -> Optional[dict]:
    """Fetch poster, rating, runtime from TMDB for a single AI recommendation."""
    headers = {"Authorization": f"Bearer {tmdb_token}", "Accept": "application/json"}
    params = {"language": tmdb_lang}
    try:
        clean_title = rec["title"].replace('"', '').replace("'", "").strip()
        search = requests.get(
            f"{TMDB_BASE}/search/movie",
            params={"query": clean_title, "year": rec["year"], **params},
            headers=headers,
            timeout=8,
        )
        results = search.json().get("results", [])
        if not results:
            return None

        movie_id = results[0]["id"]
        detail = requests.get(
            f"{TMDB_BASE}/movie/{movie_id}",
            params=params,
            headers=headers,
            timeout=8,
        )
        data = detail.json()
        data["ai_reason"] = rec["reason"]
        return data
    except Exception:
        return None


@app.post("/api/ai/recommend/")
def get_ai_recommendations(answers: QuizAnswers):
    try:
        reason_language = LANGUAGE_NAMES.get(answers.ui_language or "en", "English")
        tmdb_lang = TMDB_LANG.get(answers.ui_language or "en", "en-US")

        user_prompt = f"""
        User Quiz Results:
        - Current Mood/Context: {answers.mood}
        - Available Time: {answers.timing}
        - Preferred Language Environment: {answers.language}
        - Custom User Wishes: {answers.custom_wish}

        Select exactly 3 ideal movies. Write the 'reason' field strictly in {reason_language}.
        """

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
        movies = tool_block.input.get("movies", [])

        # Enrich all 3 movies in parallel — 3 threads, each doing search + details
        enriched = []
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = {executor.submit(enrich_with_tmdb, m, tmdb_lang): m for m in movies}
            for future in as_completed(futures):
                result = future.result()
                if result:
                    enriched.append(result)

        return enriched

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
