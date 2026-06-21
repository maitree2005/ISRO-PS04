from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi import APIRouter

from .routers import pipeline, graph, predict, criticality, simulate, models

app = FastAPI(title="Route Resilience API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(pipeline.router, prefix="/api")
app.include_router(graph.router, prefix="/api")
app.include_router(predict.router, prefix="/api")
app.include_router(criticality.router, prefix="/api")
app.include_router(simulate.router, prefix="/api")
app.include_router(models.router, prefix="/api/models")


@app.get("/api/health")
def health():
    return {"status": "ok"}
