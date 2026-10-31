import pytest
from fastapi.testclient import TestClient

from p24.api import create_app
from p24.bundle import load_bundle, seal_bundle
from p24.config import StudyConfig


@pytest.fixture
def app_client(tiny_base, tmp_path):
    bundle = tmp_path / "bundle"
    seal_bundle(tiny_base, bundle, StudyConfig())
    app = create_app(bundle)
    with TestClient(app) as client:
        yield client, bundle


@pytest.mark.parametrize(
    "body",
    [
        {"text": 123},
        {"text": None},
        {"text": "hello", "extra": True},
        {"text": " "},
        {"text": ""},
        {"text": "x" * 2049},
    ],
)
def test_invalid_input_produces_no_prediction(app_client, body):
    client, _ = app_client
    response = client.post("/predict", json=body)
    assert response.status_code in (400, 422)
    assert "model_id" not in response.json()


def test_malformed_json_is_rejected(app_client):
    client, _ = app_client
    response = client.post(
        "/predict", content=b'{"text":', headers={"content-type": "application/json"}
    )
    assert response.status_code in (400, 422)


def test_raw_character_boundary_and_public_contract(app_client):
    client, _ = app_client
    response = client.post("/predict", json={"text": "x" * 2048})
    assert response.status_code == 200
    assert set(response.json()) == {"label", "model_id", "truncated"}
    assert response.json()["model_id"] == client.get("/health").json()["model_id"]


def test_token_limit_includes_special_tokens(app_client):
    client, bundle = app_client
    predictor = load_bundle(bundle)
    for count, truncated in ((62, False), (63, True)):
        text = " ".join(["hello"] * count)
        offline = predictor.predict([text])[0]
        response = client.post("/predict", json={"text": text}).json()
        assert len(offline.token_ids) == 64
        assert offline.token_ids == (2, *([5] * 62), 3)
        assert offline.truncated is truncated
        assert response["truncated"] is truncated
        assert response["label"] == offline.label


def test_unicode_normalization_has_real_route_token_parity(app_client, monkeypatch):
    client, bundle = app_client
    expected = load_bundle(bundle).predict(["  CAFE\u0301  "])[0]
    observed = []
    predictor = client.app.state.predictor
    original = predictor.predict

    def capture(texts):
        result = original(texts)
        observed.extend(result)
        return result

    monkeypatch.setattr(predictor, "predict", capture)
    response = client.post("/predict", json={"text": "  CAFE\u0301  "})
    assert observed[0].token_ids == expected.token_ids == (2, 15, 3)
    assert response.json()["label"] == expected.label


@pytest.mark.parametrize("mode", ["tampered", "missing"])
def test_tampered_bundle_prevents_app_startup(tiny_base, tmp_path, mode):
    bundle = tmp_path / "bundle"
    seal_bundle(tiny_base, bundle, StudyConfig())
    if mode == "tampered":
        (bundle / "vocab.txt").write_text("changed", encoding="utf-8")
    else:
        (bundle / "model.safetensors").unlink()
    with pytest.raises(ValueError, match="digest|size|inventory"), TestClient(create_app(bundle)):
        pytest.fail("Invalid bundle must never become ready")
