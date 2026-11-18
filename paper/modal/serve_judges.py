"""
Modal app: GPU-hosted post-hoc JUDGES (classifiers), OpenAI-compatible.

  modal deploy paper/modal/serve_judges.py

  harmbench_cls  cais/HarmBench-Llama-2-13b-cls   (primary judge; /v1/completions, raw prompt)
  llamaguard3    meta-llama/Llama-Guard-3-8B       (/v1/chat/completions, built-in guard template)

URLs: https://<workspace>--paper-judges-harmbench-cls.modal.run/v1  (etc.)
Put them in .env as JUDGE_URL_HARMBENCH / JUDGE_URL_LLAMAGUARD, or rely on
MODAL_WORKSPACE (paper/eval/judges.yaml derives them).

Same secrets as serve_targets.py (huggingface-secret, paper-vllm-key).
These only need to be up while `paper/eval/judge_panel.py judge` runs; they
scale to zero afterwards.
"""
import os
import subprocess

import modal

VLLM_VERSION = "0.10.1.1"
JUDGES = {
    # 13B bf16 = 26 GB -> needs a 48 GB card
    "harmbench_cls": ("cais/HarmBench-Llama-2-13b-cls", "L40S", ["--max-model-len", "4096"]),
    "llamaguard3":   ("meta-llama/Llama-Guard-3-8B",    "L4",   ["--max-model-len", "8192"]),
}

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(f"vllm=={VLLM_VERSION}", "huggingface_hub[hf_transfer]")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "VLLM_USE_V1": "1"})
)
hf_cache = modal.Volume.from_name("paper-hf-cache", create_if_missing=True)
vllm_cache = modal.Volume.from_name("paper-vllm-cache", create_if_missing=True)
app = modal.App("paper-judges")

COMMON = dict(
    image=image,
    scaledown_window=10 * 60,
    timeout=24 * 60 * 60,
    max_containers=1,
    volumes={"/root/.cache/huggingface": hf_cache, "/root/.cache/vllm": vllm_cache},
    secrets=[modal.Secret.from_name("huggingface-secret"), modal.Secret.from_name("paper-vllm-key")],
)


def _launch(key):
    model, _gpu, extra = JUDGES[key]
    cmd = ["vllm", "serve", model, "--served-model-name", model, "--host", "0.0.0.0", "--port", "8000",
           "--api-key", os.environ["VLLM_API_KEY"], "--gpu-memory-utilization", "0.90",
           "--dtype", "bfloat16", "--seed", "0", "--disable-log-requests"] + extra
    subprocess.Popen(cmd)


@app.function(gpu=JUDGES["harmbench_cls"][1], **COMMON)
@modal.concurrent(max_inputs=64)
@modal.web_server(port=8000, startup_timeout=20 * 60)
def harmbench_cls():
    _launch("harmbench_cls")


@app.function(gpu=JUDGES["llamaguard3"][1], **COMMON)
@modal.concurrent(max_inputs=64)
@modal.web_server(port=8000, startup_timeout=20 * 60)
def llamaguard3():
    _launch("llamaguard3")
