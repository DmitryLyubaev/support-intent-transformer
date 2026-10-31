import argparse
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from p24.bundle import load_bundle
from p24.data import normalize


class PredictRequest(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")
    text: str = Field(min_length=1, max_length=2048)


def create_app(bundle_dir: Path) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app):
        app.state.predictor = load_bundle(bundle_dir)
        yield

    app = FastAPI(title="Support intent study", version="0.1.0", lifespan=lifespan)

    @app.get("/health")
    def health():
        return {"status": "ready", "model_id": app.state.predictor.identity.model_id}

    @app.post("/predict")
    def predict(body: PredictRequest):
        if not normalize(body.text):
            raise HTTPException(
                status_code=422, detail="text must contain non-whitespace characters"
            )
        result = app.state.predictor.predict([body.text])[0]
        return {
            "label": result.label,
            "model_id": app.state.predictor.identity.model_id,
            "truncated": result.truncated,
        }

    return app


def main():
    parser = argparse.ArgumentParser(description="Serve a verified local bundle on loopback")
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8024)
    args = parser.parse_args()
    uvicorn.run(create_app(args.bundle), host="127.0.0.1", port=args.port, workers=1)


if __name__ == "__main__":
    main()
