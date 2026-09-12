from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.api.inference import RecommendationEngine
from src.utils import load_config

engine: RecommendationEngine


class RecommendRequest(BaseModel):
    user_id: str
    top_k: int = Field(default=20, ge=1, le=200)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine
    cfg = load_config()
    engine = RecommendationEngine(cfg)
    yield
    engine = None


app = FastAPI(title="Two-Stage RecSys", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/recommend")
def recommend(req: RecommendRequest):
    if req.user_id not in engine.known_users:
        raise HTTPException(status_code=404, detail="Unknown user")
    return engine.recommend(req.user_id, req.top_k)


# The demo shares the existing model and leaves /recommend's benchmark path unchanged.
from pathlib import Path
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from src.api.demo import demo_result

static_dir = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=static_dir), name="static")


@app.get("/", include_in_schema=False)
def demo_page():
    return FileResponse(static_dir / "index.html")


@app.get("/demo/users")
def demo_users():
    users = sorted(engine.known_users)
    # Evenly spread deterministic examples across the user vocabulary.
    step = max(1, len(users) // 30)
    return {"users": users[::step][:30], "catalog_size": len(engine.id_to_raw_item)}


@app.get("/demo/recommendations")
def demo_recommendations(user_id: str):
    if user_id not in engine.known_users:
        raise HTTPException(status_code=404, detail="Unknown user")
    return demo_result(engine, load_config(), user_id)
