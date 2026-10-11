# main.py

from contextlib import asynccontextmanager

from fastapi import FastAPI

from database import (
    Base,
    engine,
)

from api.v1.conversations import router as conversations_router
from api.v1.messages import router as messages_router

# نسخه تمیز قبلی Document API
from api.v1.documents import router as documents_router


@asynccontextmanager
async def lifespan(app: FastAPI):

    Base.metadata.create_all(
        bind=engine
    )

    yield


app = FastAPI(
    title="Enterprise Assistant RAG API",
    lifespan=lifespan,
)


# --------------------------------------------------
# Routers
# --------------------------------------------------

app.include_router(
    conversations_router,
)

app.include_router(
    messages_router,
)

app.include_router(
    documents_router,
)


# --------------------------------------------------
# Health
# --------------------------------------------------

@app.get("/health")
async def health():

    return {
        "status": "healthy"
    }