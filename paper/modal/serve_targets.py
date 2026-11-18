"""
Modal app: one OpenAI-compatible vLLM endpoint per TARGET model.

  modal deploy paper/modal/serve_targets.py

Each function below becomes its own URL:
  https://<workspace>--paper-targets-<fn-with-dashes>.modal.run/v1
e.g.  https://myname--paper-targets-llama31-8b.modal.run/v1
`modal deploy` prints every URL; put them in .env as TARGET_URL_<KEY>=... (or
just set MODAL_WORKSPACE and run_matrix.py derives them).

One-time Modal secrets (see paper/SERVER_HANDOFF.md §2):
  modal secret create huggingface-secret HF_TOKEN=hf_xxx        # gated: Llama, Gemma, Mistral
  modal secret create paper-vllm-key   VLLM_API_KEY=<random>    # = MODAL_TARGET_API_KEY in .env

Cost control:
  * scale-to-zero: a container stops SCALEDOWN_S after the last request
  * max_containers=1: never more than one GPU per target
  * run_matrix.py runs targets one at a time (--targets ...), so normally only
    ONE target container is up at any moment.
  * `modal app stop paper-targets` kills everything immediately.
"""
import os
import subprocess

import modal

APP_NAME = "paper-targets"
VLLM_VERSION = "0.10.1.1"
SCALEDOWN_S = 15 * 60          # idle minutes before the GPU is released
MAX_MODEL_LEN = 4096           # longest GAP prompt (~500 tok) + 512 new tokens fits easily
MAX_CONCURRENT = 32            # requests in flight per container

# key -> (HF id, GPU, extra vllm args). Keys MUST match paper/experiments.yaml targets.*.modal_fn
TARGETS = {
    "llama31_8b":   ("meta-llama/Llama-3.1-8B-Instruct",     "L4",   []),
    "gemma2_9b":    ("google/gemma-2-9b-it",                 "L40S", []),   # 18.5 GB weights: L4 too tight
    "qwen25_7b":    ("Qwen/Qwen2.5-7B-Instruct",             "L4",   []),
    "ministral_8b": ("mistralai/Ministral-8B-Instruct-2410", "L4",
                     ["--tokenizer-mode", "mistral", "--config-format", "mistral", "--load-format", "mistral"]),
}

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(f"vllm=={VLLM_VERSION}", "huggingface_hub[hf_transfer]", "mistral_common>=1.8.0")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "VLLM_USE_V1": "1"})
)
hf_cache = modal.Volume.from_name("paper-hf-cache", create_if_missing=True)
vllm_cache = modal.Volume.from_name("paper-vllm-cache", create_if_missing=True)

app = modal.App(APP_NAME)

COMMON = dict(
    image=image,
    scaledown_window=SCALEDOWN_S,
    timeout=24 * 60 * 60,
    max_containers=1,
    volumes={"/root/.cache/huggingface": hf_cache, "/root/.cache/vllm": vllm_cache},
    secrets=[modal.Secret.from_name("huggingface-secret"), modal.Secret.from_name("paper-vllm-key")],
)


def _launch(key: str):
    model, _gpu, extra = TARGETS[key]
    cmd = [
        "vllm", "serve", model,
        "--served-model-name", model,          # clients send exactly the HF id
        "--host", "0.0.0.0", "--port", "8000",
        "--api-key", os.environ["VLLM_API_KEY"],
        "--max-model-len", str(MAX_MODEL_LEN),
        "--gpu-memory-utilization", "0.90",
        "--dtype", "bfloat16",
        "--seed", "0",
        "--disable-log-requests",
    ] + extra
    print("launching:", " ".join(c if c != os.environ["VLLM_API_KEY"] else "***" for c in cmd), flush=True)
    subprocess.Popen(cmd)


def _fn(key):
    return dict(COMMON, gpu=TARGETS[key][1])


@app.function(**_fn("llama31_8b"))
@modal.concurrent(max_inputs=MAX_CONCURRENT)
@modal.web_server(port=8000, startup_timeout=20 * 60)
def llama31_8b():
    _launch("llama31_8b")


@app.function(**_fn("gemma2_9b"))
@modal.concurrent(max_inputs=MAX_CONCURRENT)
@modal.web_server(port=8000, startup_timeout=20 * 60)
def gemma2_9b():
    _launch("gemma2_9b")


@app.function(**_fn("qwen25_7b"))
@modal.concurrent(max_inputs=MAX_CONCURRENT)
@modal.web_server(port=8000, startup_timeout=20 * 60)
def qwen25_7b():
    _launch("qwen25_7b")


@app.function(**_fn("ministral_8b"))
@modal.concurrent(max_inputs=MAX_CONCURRENT)
@modal.web_server(port=8000, startup_timeout=20 * 60)
def ministral_8b():
    _launch("ministral_8b")


@app.local_entrypoint()
def smoke(target: str = "llama31_8b"):
    """modal run paper/modal/serve_targets.py --target llama31_8b  -> one chat completion."""
    import json
    import urllib.request

    fn = {"llama31_8b": llama31_8b, "gemma2_9b": gemma2_9b,
          "qwen25_7b": qwen25_7b, "ministral_8b": ministral_8b}[target]
    url = fn.get_web_url().rstrip("/") + "/v1/chat/completions"
    key = os.environ.get("MODAL_TARGET_API_KEY") or input("VLLM_API_KEY: ")
    body = {"model": TARGETS[target][0], "messages": [{"role": "user", "content": "Say hi in 3 words."}],
            "max_tokens": 16, "temperature": 0}
    req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                 headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    print("POST", url, "(cold start can take several minutes)")
    with urllib.request.urlopen(req, timeout=1800) as r:
        print(json.load(r)["choices"][0]["message"]["content"])
