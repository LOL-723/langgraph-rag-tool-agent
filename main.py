from core.runtime import ensure_project_runtime

ensure_project_runtime()

from fastapi import FastAPI

from api.routes_agent import router as agent_router
from api.routes_llm import router as llm_router
from api.routes_rag import router as rag_router

app = FastAPI(title="LLM Client Service")

app.include_router(agent_router)
app.include_router(llm_router)
app.include_router(rag_router)


@app.get("/health")
def health():
    return {"status": "ok"}
