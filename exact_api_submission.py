"""
EXACT 2026 submission API wrapper.

Purpose:
- Expose ONE competition endpoint: POST /predict
- Return EXACT schema: a JSON list with one result object
- Reuse your current physics pipeline:
      from xai_physics.llm.schema_pipeline import solve_problem_with_llm
- Use a local Ollama server as the LLM backend.

Expected local services:
- This API: http://localhost:8000/predict
- Ollama server: http://localhost:11434

Run:
    python exact_api_submission.py

Environment variables you may set:
    EXACT_API_PORT=8000
    OLLAMA_BASE_URL=http://localhost:11434
    OLLAMA_MODEL=qwen2.5:7b-instruct
    EXACT_K=2
    REQUIRE_API_KEY=0
    EXACT_API_KEY=xai-demo-key-123
    ENABLE_TYPE1_LLM_FALLBACK=0
"""

import asyncio
import json
import math
import os
import re
import sys
import time
import traceback
from typing import Any, Dict, List, Optional, Tuple

import requests
import uvicorn
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field


# -----------------------------------------------------------------------------
# Import path setup
# -----------------------------------------------------------------------------

POSSIBLE_SRC_DIRS = [
    "/kaggle/working",
    "/kaggle/working/AI_Logic_EXACT",
    "/kaggle/working/xai_physics",
    os.getcwd(),
]

for p in POSSIBLE_SRC_DIRS:
    if os.path.exists(p) and p not in sys.path:
        sys.path.insert(0, p)

try:
    from xai_physics.llm.schema_pipeline import solve_problem_with_llm
    PHYSICS_IMPORT_ERROR = None
except Exception:
    solve_problem_with_llm = None
    PHYSICS_IMPORT_ERROR = traceback.format_exc()


# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------

API_HOST = os.getenv("EXACT_API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("EXACT_API_PORT", "8000"))

# Do NOT require API key by default. The official guide only specifies JSON POST.
# Turn this on only if the organizers explicitly agree to send your header.
REQUIRE_API_KEY = os.getenv("REQUIRE_API_KEY", "0") == "1"
EXACT_API_KEY = os.getenv("EXACT_API_KEY", "xai-demo-key-123")

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:7b-instruct")
OLLAMA_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT", "55"))
OLLAMA_NUM_CTX = int(os.getenv("OLLAMA_NUM_CTX", "4096"))
OLLAMA_NUM_PREDICT = int(os.getenv("OLLAMA_NUM_PREDICT", "512"))
OLLAMA_TEMPERATURE = float(os.getenv("OLLAMA_TEMPERATURE", "0.0"))

EXACT_K = int(os.getenv("EXACT_K", "2"))
ENABLE_TYPE1_LLM_FALLBACK = os.getenv("ENABLE_TYPE1_LLM_FALLBACK", "0") == "1"

REQUEST_TIMEOUT_SEC = float(os.getenv("REQUEST_TIMEOUT_SEC", "59"))

START_TIME = time.time()
busy_lock = asyncio.Lock()

app = FastAPI(title="EXACT 2026 Submission API", version="1.0")


# -----------------------------------------------------------------------------
# Request/response schema
# -----------------------------------------------------------------------------

class PredictRequest(BaseModel):
    query_id: str
    type: str = Field(..., pattern="^(type1|type2)$")
    query: str
    premises: List[str]
    options: List[str]


# -----------------------------------------------------------------------------
# Ollama client matching your old .generate(prompt) interface
# -----------------------------------------------------------------------------

class OllamaNotebookClient:
    """
    Drop-in client for your existing pipeline.

    Your existing pipeline expects:
        client.generate(prompt: str) -> str

    This calls Ollama's native /api/generate endpoint with format=json,
    matching the notebook code you used for CSV evaluation.
    """

    def __init__(
        self,
        model: str,
        base_url: str = OLLAMA_BASE_URL,
        temperature: float = OLLAMA_TEMPERATURE,
        num_ctx: int = OLLAMA_NUM_CTX,
        num_predict: int = OLLAMA_NUM_PREDICT,
        timeout: int = OLLAMA_TIMEOUT,
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature
        self.num_ctx = num_ctx
        self.num_predict = num_predict
        self.timeout = timeout

    def generate(self, prompt: str) -> str:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "options": {
                "temperature": self.temperature,
                "num_ctx": self.num_ctx,
                "num_predict": self.num_predict,
            },
        }
        r = requests.post(
            f"{self.base_url}/api/generate",
            json=payload,
            timeout=self.timeout,
        )
        r.raise_for_status()
        data = r.json()
        return data.get("response", "")


LLM_CLIENT = OllamaNotebookClient(model=OLLAMA_MODEL)


# -----------------------------------------------------------------------------
# Unit and answer normalization helpers
# -----------------------------------------------------------------------------

NUMBER_RE = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?")

UNIT_ALIASES = {
    "": "",
    "Ω": "ohm",
    "Ω": "ohm",
    "ohms": "ohm",
    "Ohm": "ohm",
    "μ": "u",
    "µ": "u",
    "μF": "uF",
    "µF": "uF",
    "μC": "uC",
    "µC": "uC",
    "μA": "uA",
    "µA": "uA",
    "μJ": "uJ",
    "µJ": "uJ",
    "μH": "uH",
    "µH": "uH",
    "μT": "uT",
    "µT": "uT",
    "μWb": "uWb",
    "µWb": "uWb",
    "m³": "m^3",
    "cm³": "cm^3",
    "mm³": "mm^3",
    "m²": "m^2",
    "cm²": "cm^2",
    "mm²": "mm^2",
    "J/m³": "J/m^3",
    "J/m²": "J/m^2",
    "°": "deg",
    "°C": "degC",
}

KNOWN_UNITS = [
    "V/m", "N/C", "J/m^3", "J/m3", "rad/s", "turns/m",
    "kohm", "Mohm", "ohm",
    "pF", "nF", "uF", "mF", "F",
    "pC", "nC", "uC", "mC", "C",
    "uA", "mA", "A",
    "mV", "kV", "V",
    "uJ", "mJ", "nJ", "pJ", "J",
    "uW", "mW", "W",
    "uH", "mH", "H",
    "uT", "mT", "T",
    "uWb", "mWb", "Wb",
    "kHz", "Hz",
    "ms", "s",
    "cm^2", "mm^2", "m^2",
    "cm", "mm", "um", "nm", "m",
    "kg", "g",
    "%", "times", "x", "deg", "degC",
]
KNOWN_UNITS = sorted(set(KNOWN_UNITS), key=len, reverse=True)


def normalize_unit_text(unit: Any) -> str:
    if unit is None:
        return ""
    try:
        if math.isnan(unit):
            return ""
    except TypeError:
        pass

    u = str(unit).strip()
    if not u or u.lower() == "nan":
        return ""

    for old, new in UNIT_ALIASES.items():
        if old:
            u = u.replace(old, new)

    u = u.replace("^2", "^2").replace("^3", "^3")
    u = re.sub(r"\s*/\s*", "/", u)
    u = re.sub(r"\s+", "", u)
    return u


def strip_unit_from_answer(answer: Any, explicit_unit: Any = None) -> Tuple[str, str]:
    """
    Competition Type 2 wants:
        answer: numerical value only
        unit: ASCII unit only

    Your old solver may return answer as "5 A" or unit separately. This normalizes both.
    """
    ans = "" if answer is None else str(answer).strip()
    unit = normalize_unit_text(explicit_unit)

    if unit:
        # If answer ends with the same unit, remove it.
        ans = re.sub(rf"\s*{re.escape(str(explicit_unit).strip())}\s*$", "", ans).strip()
        ans = re.sub(rf"\s*{re.escape(unit)}\s*$", "", ans).strip()
        return ans, unit

    # Convert common Unicode units before regex extraction.
    clean = normalize_text_for_problem(ans)

    # Try pattern: final number + unit at the end.
    # Example: "5 A", "7.5 ohm", "2.3e-6 C".
    for u in KNOWN_UNITS:
        m = re.search(rf"({NUMBER_RE.pattern})\s*{re.escape(u)}\s*$", clean)
        if m:
            return m.group(1), normalize_unit_text(u)

    # Try extracting a number and a known unit anywhere; useful for "I = 5 A".
    nums = list(NUMBER_RE.finditer(clean))
    if nums:
        last_num = nums[-1]
        tail = clean[last_num.end():].strip()
        for u in KNOWN_UNITS:
            if tail.startswith(u):
                return last_num.group(0), normalize_unit_text(u)

    return ans, ""


def normalize_text_for_problem(text: Any) -> str:
    """Light notation cleanup before sending to your physics parser."""
    if text is None:
        return ""
    s = str(text)

    replacements = {
        "\\times": "*",
        "\\cdot": "*",
        "\\div": "/",
        "\\Omega": "ohm",
        "Ω": "ohm",
        "Ω": "ohm",
        "\\mu": "u",
        "μ": "u",
        "µ": "u",
        "\\%": "%",
        "\\degree": "deg",
        "°": "deg",
        "\\leq": "<=",
        "\\geq": ">=",
        "\\neq": "!=",
        "\\approx": "~",
        "\\pm": "+/-",
        "\\mp": "-/+",
    }
    for old, new in replacements.items():
        s = s.replace(old, new)

    # Basic LaTeX fraction normalization: \frac{a}{b} -> (a)/(b)
    s = re.sub(r"\\frac\{([^{}]+)\}\{([^{}]+)\}", r"(\1)/(\2)", s)

    # Basic scientific notation: 3 * 10^{-6} -> 3e-6
    s = re.sub(
        r"(\d+(?:\.\d+)?)\s*\*\s*10\s*\^\s*\{?\s*([-+]?\d+)\s*\}?",
        r"\1e\2",
        s,
    )

    s = re.sub(r"\s+", " ", s).strip()
    return s


def make_nonempty_explanation(answer: str, unit: str, source_status: str = "") -> str:
    display = f"{answer} {unit}".strip()
    if source_status:
        return f"Computed by the physics pipeline. Final answer: {display}. Solver status: {source_status}."
    return f"Computed by the physics pipeline. Final answer: {display}."


def safe_json_dumps(obj: Any, max_chars: int = 1200) -> str:
    try:
        s = json.dumps(obj, ensure_ascii=False)
    except Exception:
        s = str(obj)
    if len(s) > max_chars:
        s = s[:max_chars] + "..."
    return s


# -----------------------------------------------------------------------------
# Type 2: physics pipeline wrapper
# -----------------------------------------------------------------------------


def solve_type2_physics(req: PredictRequest) -> Dict[str, Any]:
    if solve_problem_with_llm is None:
        return {
            "query_id": req.query_id,
            "answer": "0",
            "unit": "",
            "explanation": "Physics pipeline import failed. Check that xai_physics is available in the notebook path.",
            "premises_used": [],
            "reasoning": {
                "type": "error",
                "steps": [PHYSICS_IMPORT_ERROR or "Unknown import error"],
            },
        }

    problem = normalize_text_for_problem(req.query)
    output = solve_problem_with_llm(problem, LLM_CLIENT, k=EXACT_K)
    result = output.solve_result

    raw_answer = getattr(result, "answer", None)
    raw_unit = getattr(result, "unit", "")
    answer, unit = strip_unit_from_answer(raw_answer, raw_unit)

    # If solver keeps unit only inside answer, strip_unit_from_answer will recover it.
    status = str(getattr(result, "status", ""))
    error = getattr(result, "error", None)
    domain = getattr(result, "domain", None)

    explanation = getattr(result, "explanation", None)
    if not explanation:
        explanation = make_nonempty_explanation(answer, unit, status)
        if error:
            explanation += f" Note: internal solver message: {str(error)[:250]}"

    steps = []
    if domain:
        steps.append(f"Detected domain: {domain}")
    if status:
        steps.append(f"Solver status: {status}")

    schema = getattr(output, "schema", None)
    if schema is not None:
        steps.append("Structured extraction/schema: " + safe_json_dumps(schema))

    if error:
        steps.append("Internal solver message: " + str(error)[:500])

    if not steps:
        steps = ["Physics pipeline returned the final answer."]

    return {
        "query_id": req.query_id,
        "answer": str(answer).strip() or "0",
        "unit": normalize_unit_text(unit),
        "explanation": str(explanation).strip() or make_nonempty_explanation(answer, unit, status),
        "premises_used": [],
        "reasoning": {
            "type": "cot",
            "steps": steps,
        },
    }


# -----------------------------------------------------------------------------
# Type 1: temporary fallback because logic pipeline is not implemented yet
# -----------------------------------------------------------------------------


def normalize_answer_to_option(answer: Any, options: List[str]) -> str:
    ans = str(answer).strip()
    if not options:
        return ans

    for opt in options:
        if ans == opt:
            return opt
    for opt in options:
        if ans.lower() == opt.lower():
            return opt
    for opt in options:
        if opt.lower() in ans.lower():
            return opt
    if "Uncertain" in options:
        return "Uncertain"
    return options[0]


def solve_type1_fallback(req: PredictRequest) -> Dict[str, Any]:
    # Safe default. This will not score well for Type 1, but keeps schema valid.
    if req.options:
        answer = "Uncertain" if "Uncertain" in req.options else req.options[0]
    else:
        answer = "Uncertain"

    return {
        "query_id": req.query_id,
        "answer": normalize_answer_to_option(answer, req.options),
        "unit": "",
        "explanation": "Logic Type 1 pipeline is not implemented in this submission wrapper yet; returning a schema-valid fallback answer.",
        "premises_used": [],
        "reasoning": {
            "type": "fol",
            "steps": [
                "Type 1 request received.",
                "No FOL/logic solver is connected yet.",
                "Returned fallback answer to preserve the required JSON schema.",
            ],
        },
    }


def solve_type1_llm_fallback(req: PredictRequest) -> Dict[str, Any]:
    """Optional last-resort Type 1 LLM fallback. Disabled by default."""
    prompt = f"""
You are answering an educational logic question.
Return ONLY valid JSON with keys: answer, premises_used, explanation, steps.

Rules:
- If options is non-empty, answer must be exactly one of the provided options.
- premises_used must be a list of 0-based integer indices from the premises.

query: {req.query}
options: {json.dumps(req.options, ensure_ascii=False)}
premises:
{json.dumps(req.premises, ensure_ascii=False, indent=2)}
""".strip()

    try:
        raw = LLM_CLIENT.generate(prompt)
        data = json.loads(raw)
        answer = normalize_answer_to_option(data.get("answer", "Uncertain"), req.options)
        used = data.get("premises_used", [])
        if not isinstance(used, list):
            used = []
        used = sorted({int(i) for i in used if isinstance(i, int) or str(i).isdigit() if 0 <= int(i) < len(req.premises)})
        steps = data.get("steps", [])
        if not isinstance(steps, list):
            steps = [str(steps)]
        explanation = str(data.get("explanation", "Answered by LLM fallback.")).strip()
        return {
            "query_id": req.query_id,
            "answer": answer,
            "unit": "",
            "explanation": explanation or "Answered by LLM fallback.",
            "premises_used": used,
            "reasoning": {"type": "fol", "steps": steps},
        }
    except Exception as e:
        out = solve_type1_fallback(req)
        out["explanation"] += f" LLM fallback failed: {str(e)[:200]}"
        return out


# -----------------------------------------------------------------------------
# Final schema normalization
# -----------------------------------------------------------------------------


def normalize_result(req: PredictRequest, result: Dict[str, Any]) -> Dict[str, Any]:
    result = dict(result or {})
    result["query_id"] = req.query_id

    if req.type == "type1":
        result["unit"] = ""
        result["answer"] = normalize_answer_to_option(result.get("answer", "Uncertain"), req.options) if req.options else str(result.get("answer", "Uncertain")).strip()

        used = result.get("premises_used", [])
        if not isinstance(used, list):
            used = []
        clean_used = []
        for x in used:
            try:
                ix = int(x)
                if 0 <= ix < len(req.premises):
                    clean_used.append(ix)
            except Exception:
                pass
        result["premises_used"] = sorted(set(clean_used))

    elif req.type == "type2":
        answer, unit_from_answer = strip_unit_from_answer(result.get("answer", ""), result.get("unit", ""))
        result["answer"] = str(answer).strip() or "0"
        result["unit"] = normalize_unit_text(result.get("unit", "") or unit_from_answer)
        result["premises_used"] = []

    explanation = str(result.get("explanation", "")).strip()
    if not explanation:
        explanation = "The system produced this answer using the configured pipeline."
    result["explanation"] = explanation

    if "reasoning" not in result:
        result["reasoning"] = None

    # Keep exactly the official fields.
    return {
        "query_id": str(result["query_id"]),
        "answer": str(result["answer"]),
        "unit": str(result["unit"]),
        "explanation": str(result["explanation"]),
        "premises_used": result["premises_used"],
        "reasoning": result["reasoning"],
    }


async def run_prediction(req: PredictRequest) -> Dict[str, Any]:
    if req.type == "type2":
        return await asyncio.to_thread(solve_type2_physics, req)

    if req.type == "type1" and ENABLE_TYPE1_LLM_FALLBACK:
        return await asyncio.to_thread(solve_type1_llm_fallback, req)

    return solve_type1_fallback(req)


# -----------------------------------------------------------------------------
# Routes
# -----------------------------------------------------------------------------

@app.get("/")
def root():
    return {
        "name": "EXACT 2026 Submission API",
        "prediction_endpoint": "/predict",
        "ollama_base_url": OLLAMA_BASE_URL,
        "ollama_model": OLLAMA_MODEL,
    }


@app.get("/health")
def health():
    return {
        "status": "ok" if solve_problem_with_llm is not None else "error",
        "physics_pipeline_imported": solve_problem_with_llm is not None,
        "physics_import_error": PHYSICS_IMPORT_ERROR,
        "ollama_base_url": OLLAMA_BASE_URL,
        "ollama_model": OLLAMA_MODEL,
        "type1_mode": "llm_fallback" if ENABLE_TYPE1_LLM_FALLBACK else "schema_fallback",
        "uptime_sec": round(time.time() - START_TIME, 2),
    }


@app.get("/status")
def status():
    return {
        "status": "running",
        "busy": busy_lock.locked(),
        "mode": "single-request",
        "uptime_sec": round(time.time() - START_TIME, 2),
    }


@app.get("/ollama_tags_proxy")
def ollama_tags_proxy():
    """Convenience check only. For urls.txt, expose the real Ollama URL through its own tunnel."""
    r = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=10)
    r.raise_for_status()
    return r.json()


@app.get("/ollama_v1_models_proxy")
def ollama_v1_models_proxy():
    """Convenience check for Ollama's OpenAI-compatible /v1/models endpoint."""
    r = requests.get(f"{OLLAMA_BASE_URL}/v1/models", timeout=10)
    r.raise_for_status()
    return r.json()


@app.post("/predict")
async def predict(req: PredictRequest, x_api_key: Optional[str] = Header(default=None)):
    if REQUIRE_API_KEY and x_api_key != EXACT_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")

    if busy_lock.locked():
        # The guide says queries are sequential. This protects you from accidental overlap.
        raise HTTPException(status_code=429, detail="Model is busy. Please retry in a few seconds.")

    async with busy_lock:
        t0 = time.time()
        try:
            result = await asyncio.wait_for(run_prediction(req), timeout=REQUEST_TIMEOUT_SEC)
            result = normalize_result(req, result)
            elapsed = round(time.time() - t0, 3)
            print(f"[PREDICT] {req.query_id} {req.type} {elapsed}s answer={result['answer']} unit={result['unit']}", flush=True)
            return [result]

        except Exception as e:
            elapsed = round(time.time() - t0, 3)
            print(f"[ERROR] {req.query_id} {req.type} {elapsed}s {type(e).__name__}: {e}", flush=True)
            print(traceback.format_exc(), flush=True)

            # Still return official schema as a last resort.
            if req.type == "type1":
                fallback_answer = "Uncertain" if "Uncertain" in req.options else (req.options[0] if req.options else "Uncertain")
                fallback = {
                    "query_id": req.query_id,
                    "answer": fallback_answer,
                    "unit": "",
                    "explanation": f"Internal error while processing the Type 1 query: {str(e)[:300]}",
                    "premises_used": [],
                    "reasoning": None,
                }
            else:
                fallback = {
                    "query_id": req.query_id,
                    "answer": "0",
                    "unit": "",
                    "explanation": f"Internal error while processing the physics query: {str(e)[:300]}",
                    "premises_used": [],
                    "reasoning": None,
                }
            return [normalize_result(req, fallback)]


if __name__ == "__main__":
    uvicorn.run(app, host=API_HOST, port=API_PORT, log_level="info")
