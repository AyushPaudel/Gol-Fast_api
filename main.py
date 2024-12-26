from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from pydantic import BaseModel
from typing import Optional, Dict
import uvicorn
from dotenv import load_dotenv
import os
import sqlite3
import json
from uuid import uuid4
from openai import AsyncOpenAI
from collections import defaultdict
from datetime import datetime, timedelta
from contextlib import asynccontextmanager
from fastapi.responses import JSONResponse

# Cache setup
cache = defaultdict(dict)
cache_expiry = defaultdict(dict)


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield


app = FastAPI(lifespan=lifespan)
templates = Jinja2Templates(directory="templates")
SESSION_KEY = APIKeyHeader(name="X-Session-ID", auto_error=False)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")


class GameState(BaseModel):
    player_history: list
    happiness: int
    wealth: int
    turn_count: int
    game_id: str


class GameChoice(BaseModel):
    choice: Optional[str] = None


async def get_session_id(session_id: str = Depends(SESSION_KEY)) -> str:
    return session_id or str(uuid4())


class ModernWesterosGame:
    def __init__(self):
        load_dotenv()
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key not found")

        self.client = AsyncOpenAI(api_key=api_key)
        self.player_history = []
        self.happiness = 30
        self.wealth = 100000
        self.turn_count = 0
        self.game_id = str(uuid4())

        self.setup_database()

    def get_state(self) -> GameState:
        return GameState(
            player_history=self.player_history,
            happiness=self.happiness,
            wealth=self.wealth,
            turn_count=self.turn_count,
            game_id=self.game_id,
        )

    def load_state(self, state: GameState):
        self.player_history = state.player_history
        self.happiness = state.happiness
        self.wealth = state.wealth
        self.turn_count = state.turn_count
        self.game_id = state.game_id

    def setup_database(self):
        conn = sqlite3.connect("westeros_realty.db")
        cursor = conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS choices (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                choice_text TEXT NOT NULL,
                game_id TEXT NOT NULL
            )
        """
        )
        conn.commit()
        conn.close()

    def store_choice(self, choice_text: str):
        conn = sqlite3.connect("westeros_realty.db")
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO choices (choice_text, game_id) VALUES (?, ?)",
            (choice_text, self.game_id),
        )
        conn.commit()
        conn.close()

    def get_previous_choices(self) -> list:
        conn = sqlite3.connect("westeros_realty.db")
        cursor = conn.cursor()
        cursor.execute(
            "SELECT choice_text FROM choices WHERE game_id = ?", (self.game_id,)
        )
        choices = [row[0] for row in cursor.fetchall()]
        conn.close()
        return choices

    def format_history(self) -> str:
        return (
            "No previous choices."
            if not self.player_history
            else " Then ".join(self.player_history)
        )

    async def get_story_segment(self, current_situation: str) -> str:
        try:
            previous_choices = self.get_previous_choices()
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are a witty narrator in a world where Jon Snow has traded his Night's Watch cloak for a realtor's suit. "
                        "Mix modern real estate scenarios with Game of Thrones references and humor."
                    ),
                },
                {
                    "role": "user",
                    "content": f"""
Current Status:
Happiness: {self.happiness}/100
Wealth: ${self.wealth}

Previous choices in this game: {self.format_history()}
Previously used choices in database: {json.dumps(previous_choices)}
Current situation: {current_situation}

Generate a short story segment (2-3 sentences) with Game of Thrones references.
Then provide 3 UNIQUE choices that are witty and concise (max 15 words each).
Use GOT-style humor and puns in the choices. Each choice must have exact numerical impacts in parentheses.

Format your response as follows:
STORY: [Your story text here]
CHOICES:
1. Act like a Dothraki: Take what is yours with fire and blood (Happiness: +20, Wealth: -50000)
2. Consult with Bran: See the future and make wise investments (Happiness: +10, Wealth: +25000)
3. Meet with the Iron Bank: Get a loan and rule the market (Happiness: -15, Wealth: +40000)""",
                },
            ]

            response = await self.client.chat.completions.create(
                model="gpt-4", messages=messages, temperature=0.8
            )
            return response.choices[0].message.content

        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"Error generating story segment: {str(e)}"
            )

    async def get_consequence(self, choice: str) -> str:
        try:
            messages = [
                {
                    "role": "system",
                    "content": "You are a witty narrator blending modern real estate with Game of Thrones references.",
                },
                {
                    "role": "user",
                    "content": f"""
Based on Jon's history: {self.format_history()}
Current status:
Happiness: {self.happiness}/100
Wealth: ${self.wealth}
They chose: {choice}

Generate a consequence (2-3 sentences) that blends modern real estate outcomes with Game of Thrones references.
Be creative and humorous while keeping the real estate aspects realistic.""",
                },
            ]

            response = await self.client.chat.completions.create(
                model="gpt-4", messages=messages, temperature=0.8
            )
            return response.choices[0].message.content

        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"Error generating consequence: {str(e)}"
            )

    def parse_stats_impact(self, choice: str) -> tuple[int, int]:
        try:
            if "(" not in choice or ")" not in choice:
                return 0, 0

            impacts = choice.split("(")[1].split(")")[0].split(",")
            if len(impacts) < 2:
                return 0, 0

            def extract_number(impact_str):
                return int("".join(c for c in impact_str if c.isdigit() or c in "+-"))

            happiness_impact = extract_number(impacts[0])
            wealth_impact = extract_number(impacts[1])

            return happiness_impact, wealth_impact

        except Exception as e:
            print(f"Error parsing choice impacts: {e}")
            return 0, 0

    def update_stats(self, choice_text: str) -> tuple[int, int]:
        happiness_impact, wealth_impact = self.parse_stats_impact(choice_text)
        self.happiness = max(0, min(100, self.happiness + happiness_impact))
        self.wealth = max(0, self.wealth + wealth_impact)
        return happiness_impact, wealth_impact

    def check_game_over(self) -> tuple[bool, str]:
        if self.happiness <= 0:
            return (
                True,
                "Game Over! Your happiness has reached zero. The night is dark and full of terrors.",
            )
        if self.wealth <= 0:
            return (
                True,
                "Game Over! You've gone broke. Even the Iron Bank won't help you now.",
            )
        if self.happiness >= 80 and self.wealth >= 150000:
            return (
                True,
                "Victory! You've achieved both wealth and happiness. The North remembers your success!",
            )
        if self.turn_count >= 10:
            return (
                True,
                "Game Over! You've completed your journey, but haven't reached greatness.",
            )
        return False, ""

    async def start_game(self) -> str:
        initial_situation = (
            "Jon Snow, having left the Night's Watch for a new life in modern-day real estate, "
            "stands before the gleaming Glass Tower in downtown King's Landing (formerly Manhattan). "
            "His father's words echo in his head: 'Winter is Coming... and so is the housing market crash.'"
        )
        return await self.get_story_segment(initial_situation)

    async def make_choice(self, choice_text: str) -> tuple[str, str]:
        try:
            self.turn_count += 1
            if not choice_text or len(choice_text.strip()) == 0:
                raise ValueError("Empty choice text")

            self.store_choice(choice_text)
            self.update_stats(choice_text)

            consequence = await self.get_consequence(choice_text)
            next_segment = await self.get_story_segment(consequence)

            self.player_history.append(choice_text)
            return consequence, next_segment

        except Exception as e:
            print(f"Error processing choice: {e}")
            default_segment = await self.get_story_segment(
                "Jon needs to reassess his options..."
            )
            return "The outcome of your choice was unclear...", default_segment


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    session_id = request.cookies.get("session_id", str(uuid4()))
    response = templates.TemplateResponse("index.html", {"request": request})
    response.set_cookie("session_id", session_id)
    return response


@app.post("/api/game")
async def game_action(game_choice: GameChoice, request: Request):
    session_id = request.cookies.get("session_id")
    if not session_id:
        session_id = str(uuid4())

    try:
        current_time = datetime.now()

        if session_id in cache and current_time < cache_expiry[session_id].get(
            "game_state", current_time
        ):
            game = ModernWesterosGame()
            game.load_state(GameState(**cache[session_id]["game_state"]))
            current_segment = cache[session_id]["current_segment"]

            if game_choice.choice:
                if game_choice.choice not in ["1", "2", "3"]:
                    raise HTTPException(status_code=400, detail="Invalid choice")

                choices = current_segment.split("CHOICES:\n")[1].split("\n")
                chosen_action = choices[int(game_choice.choice) - 1].lstrip("123. ")
                consequence, next_segment = await game.make_choice(chosen_action)
            else:
                next_segment = current_segment
                consequence = None
        else:
            game = ModernWesterosGame()
            current_segment = await game.start_game()
            consequence = None
            next_segment = current_segment

        expiry = current_time + timedelta(hours=1)
        cache[session_id] = {
            "game_state": game.get_state().dict(),
            "current_segment": next_segment,
        }
        cache_expiry[session_id] = {"game_state": expiry, "current_segment": expiry}

        is_game_over, message = game.check_game_over()
        response_data = {
            "status": "success",
            "session_id": session_id,
            "game_state": {
                "turn": game.turn_count + 1,
                "happiness": game.happiness,
                "wealth": game.wealth,
                "consequence": consequence,
                "story": next_segment.split("STORY:")[1].split("CHOICES:")[0].strip(),
                "choices": [
                    choice.strip()
                    for choice in next_segment.split("CHOICES:\n")[1].split("\n")
                    if choice.strip()
                ][:3],
                "is_game_over": is_game_over,
                "game_over_message": message if is_game_over else None,
            },
        }

        response = JSONResponse(content=response_data)
        response.set_cookie("session_id", session_id)
        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/reset")
async def reset_game(request: Request):
    session_id = request.cookies.get("session_id")
    if session_id and session_id in cache:
        del cache[session_id]
        del cache_expiry[session_id]
    response = JSONResponse({"status": "success", "message": "Game reset successfully"})
    response.delete_cookie("session_id")
    return response


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
