#!/usr/bin/env python3
"""Run a bounded vLLM benchmark sweep against one running Inkling variant.

Adapted from deepseek-flash-cc-lab/benchmarks/natural (glm-lab corpus)."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import signal
import statistics
import subprocess
import sys
import time
import urllib.request
import uuid
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path


DEFAULT_CONCURRENCIES = "1,8,32"
DEFAULT_REPETITIONS = 3
DEFAULT_INPUT_TOKENS = 1024
DEFAULT_OUTPUT_TOKENS = 256
DEFAULT_NATURAL_MAX_TOKENS = 1024
DEFAULT_LONG_MAX_TOKENS = 2048
DEFAULT_MIN_FINAL_TOKENS = 256
DEFAULT_BUDGET_SECONDS = 5400
DEFAULT_TRIAL_TIMEOUT_SECONDS = 900
DEFAULT_SEED = 20261007
MIN_REQUESTS = 64
REQUESTS_PER_CONCURRENCY = 8
MIN_WARMUP_REQUESTS = 16
WARMUP_REQUESTS_PER_CONCURRENCY = 2
NATURAL_MIN_REQUESTS = 16
NATURAL_REQUESTS_PER_CONCURRENCY = 1
NATURAL_MIN_WARMUP_REQUESTS = 4
NATURAL_WARMUP_REQUESTS_PER_CONCURRENCY = 1
CONCURRENCY_SEED_STRIDE = 1000
HELP_TIMEOUT_SECONDS = 60
METRICS_TIMEOUT_SECONDS = 10
TERMINATION_GRACE_SECONDS = 5
MIN_REMAINING_SECONDS = 30
METRIC_NAMES = ("output_throughput", "final_output_throughput", "request_throughput", "median_ttft_ms", "p95_ttft_ms", "median_tpot_ms", "p95_tpot_ms")
CLIENT_PACKAGES = ("vllm", "torch", "flashinfer-python", "transformers")
NATURAL_REQUEST_PREFIX = "Benchmark request"
FINISH_COUNTER_NAME = "vllm:request_success_total"
FINAL_TOKEN_COUNT_PROGRAM = """
import json
import sys
from pathlib import Path
from vllm.tokenizers import get_tokenizer

tokenizer = get_tokenizer(sys.argv[1], tokenizer_mode="inkling")
result = json.loads(Path(sys.argv[2]).read_text())
lengths = [len(tokenizer.encode(text, add_special_tokens=False)) for text in result["generated_texts"]]
Path(sys.argv[3]).write_text(json.dumps(lengths) + "\\n")
"""


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def save_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n")


def command_for(args: argparse.Namespace, trial: dict, run_id: str) -> list[str]:
    extra_body = {
        "seed": trial["seed"],
        "cache_salt": f"{run_id}-{trial['name']}",
    }
    if args.dataset_path:
        extra_body["chat_template_kwargs"] = {"reasoning_effort": "low"}
    else:
        extra_body["repetition_detection"] = {"max_pattern_size": 0, "min_pattern_size": 0, "min_count": 0}
    command = [
        sys.executable, "-m", "vllm.entrypoints.cli.main", "bench", "serve",
        "--base-url", args.base_url.rstrip("/"),
        "--backend", "openai-chat" if args.dataset_path else "openai",
        "--endpoint", "/v1/chat/completions" if args.dataset_path else "/v1/completions",
        "--model", args.model, "--tokenizer", args.tokenizer,
        "--tokenizer-mode", "inkling",
        "--max-concurrency", str(trial["concurrency"]),
        "--num-prompts", str(trial["requests"]),
        "--num-warmups", "0", "--ready-check-timeout-sec", "0",
        "--request-rate", "inf", "--temperature", "0", "--top-p", "1",
        "--seed", str(trial["seed"]),
        "--percentile-metrics", "ttft,tpot,itl,e2el", "--metric-percentiles", "50,95,99",
        "--save-result", "--save-detailed", "--disable-tqdm",
        "--result-dir", str(args.out_dir), "--result-filename", f"{trial['name']}.json",
        "--label", f"{args.variant}-{trial['name']}",
        "--extra-body", json.dumps(extra_body),
    ]
    if args.dataset_path:
        command += [
            "--dataset-name", "custom", "--dataset-path", str(args.dataset_path),
            "--custom-output-len", str(args.output_tokens), "--no-oversample",
            "--disable-shuffle", "--skip-chat-template",
        ]
    else:
        command += [
            "--ignore-eos",
            "--dataset-name", "random", "--random-input-len", str(args.input_tokens),
            "--random-output-len", str(args.output_tokens),
            "--random-prefix-len", "0", "--random-range-ratio", "0.0",
        ]
    return command


def run_command(command: list[str], log_path: Path, timeout: float) -> None:
    with log_path.open("w") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            returncode = process.wait(timeout=timeout)
        except (subprocess.TimeoutExpired, KeyboardInterrupt):
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=TERMINATION_GRACE_SECONDS)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            raise
        if returncode:
            raise RuntimeError(f"benchmark command exited {returncode}; see {log_path}")


def remaining_seconds(deadline: float, limit: float) -> float:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError("sweep budget exhausted")
    return min(remaining, limit)


def capture_metrics(base_url: str, path: Path, timeout: float) -> str:
    try:
        with urllib.request.urlopen(f"{base_url.rstrip('/')}/metrics", timeout=timeout) as response:
            body = response.read().decode("utf-8")
        path.write_text(body)
        return body
    except Exception as exc:
        path.with_suffix(".error.txt").write_text(f"{type(exc).__name__}: {exc}\n")
        raise


def finish_counts(metrics: str) -> dict[str, float]:
    from prometheus_client.parser import text_string_to_metric_families

    counts = {}
    for family in text_string_to_metric_families(metrics):
        for sample in family.samples:
            if sample.name == FINISH_COUNTER_NAME:
                reason = sample.labels["finished_reason"]
                if not math.isfinite(sample.value) or sample.value < 0:
                    raise ValueError("invalid request finish counter")
                counts[reason] = counts.get(reason, 0) + sample.value
    if not counts:
        raise ValueError("request finish counters are missing from engine metrics")
    return counts


def validate_finish_counts(before: str, after: str, requests: int, expected_reason: str) -> dict[str, float]:
    before_counts, after_counts = finish_counts(before), finish_counts(after)
    if before_counts.keys() != after_counts.keys():
        raise ValueError("request finish counter labels changed during trial")
    deltas = {reason: after_counts[reason] - value for reason, value in before_counts.items()}
    if deltas.get(expected_reason) != requests or any(delta != 0 for reason, delta in deltas.items() if reason != expected_reason):
        raise ValueError(f"request termination mismatch: expected {requests} {expected_reason}, observed {deltas}")
    return deltas


def validate_result(result: dict, requests: int, input_tokens: int | None, output_tokens: int, expected_rows: list[dict] | None = None, final_lens: list[int] | None = None, min_final_tokens: int = DEFAULT_MIN_FINAL_TOKENS) -> dict:
    if result.get("completed") != requests or result.get("failed") != 0:
        raise ValueError(f"request count mismatch: completed={result.get('completed')} failed={result.get('failed')} expected={requests}")
    if any(result.get("errors", [])):
        raise ValueError("benchmark recorded request errors")
    if not math.isfinite(result.get("duration", 0)) or result.get("duration", 0) <= 0:
        raise ValueError("benchmark duration must be finite and positive")
    if any(not isinstance(result.get(name), (int, float)) or not math.isfinite(result[name]) or result[name] <= 0 for name in METRIC_NAMES if name != "final_output_throughput"):
        raise ValueError("benchmark throughput and latency metrics must be finite and positive")
    output_lens = result.get("output_lens", [])
    if len(output_lens) != requests:
        raise ValueError("output lengths are missing for some requests")
    if input_tokens is not None and any(length != output_tokens for length in output_lens):
        raise ValueError("actual output lengths differ from the fixed output-token contract")
    if input_tokens is None and any(length <= 0 or length >= output_tokens for length in output_lens):
        raise ValueError("natural response is empty or reached its generation cap")
    input_lens = result.get("input_lens", [])
    if len(input_lens) != requests or any(length <= 0 for length in input_lens) or (input_tokens is not None and any(length != input_tokens for length in input_lens)):
        raise ValueError("actual input lengths differ from the input-token contract")
    if expected_rows is not None:
        generated = result.get("generated_texts", [])
        if len(generated) != requests or len(expected_rows) < requests:
            raise ValueError("natural response/oracle counts do not match the requests")
        for row, text in zip(expected_rows, generated):
            parsed = json.loads(text.strip())
            if json.dumps(parsed, sort_keys=True) != json.dumps(row["expected_json"], sort_keys=True):
                raise ValueError(f"natural semantic mismatch for case {row['case_id']}")
    if final_lens is not None:
        generated = result.get("generated_texts", [])
        if len(generated) != requests or len(final_lens) != requests:
            raise ValueError("long-generation response count does not match requests")
        if any(length < min_final_tokens for length in final_lens):
            raise ValueError(f"long-generation response has fewer than {min_final_tokens} final-content tokens")
        return {"final_output_lens": final_lens, "final_output_throughput": sum(final_lens) / result["duration"]}
    return {}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokenizer", required=True, help="Pinned local model-pack tokenizer path")
    parser.add_argument("--out-dir", required=True, type=Path, help="New output directory in this client's filesystem")
    parser.add_argument("--variant", required=True, help="Label only; the operator must configure the server")
    parser.add_argument("--server-id", required=True, help="Recorded node and engine-image/revision identity")
    parser.add_argument("--cc-mode", required=True, choices=("on", "on-devtools", "off", "unknown"), help="Observed mode label; no mode changes are performed")
    parser.add_argument("--concurrencies", default=DEFAULT_CONCURRENCIES)
    parser.add_argument("--repetitions", type=positive_int, default=DEFAULT_REPETITIONS)
    parser.add_argument("--start-repetition", type=positive_int, default=1)
    parser.add_argument("--input-tokens", type=positive_int, default=DEFAULT_INPUT_TOKENS)
    parser.add_argument("--output-tokens", type=positive_int, help="Synthetic length (256), oracle cap (1024), or long-generation cap (2048)")
    parser.add_argument("--seed", type=positive_int, default=DEFAULT_SEED)
    parser.add_argument("--budget-seconds", type=positive_int, default=DEFAULT_BUDGET_SECONDS)
    parser.add_argument("--trial-timeout-seconds", type=positive_int, default=DEFAULT_TRIAL_TIMEOUT_SECONDS)
    parser.add_argument("--dataset-path", type=Path, help="Optional frozen natural JSONL with prompt/case_id/expected_json; enables chat, real EOS/guards, and semantic validation")
    parser.add_argument("--natural-kind", choices=("oracle", "long-generation"), default="oracle")
    parser.add_argument("--min-final-tokens", type=positive_int, default=DEFAULT_MIN_FINAL_TOKENS, help="Required final-content tokens in the long-generation lane")
    parser.add_argument("--plan-only", action="store_true", help="Write commands and manifest without importing vLLM or sending traffic")
    args = parser.parse_args()
    deadline = time.monotonic() + args.budget_seconds
    if args.output_tokens is None:
        args.output_tokens = (DEFAULT_LONG_MAX_TOKENS if args.natural_kind == "long-generation" else DEFAULT_NATURAL_MAX_TOKENS) if args.dataset_path else DEFAULT_OUTPUT_TOKENS
    concurrencies = [positive_int(value) for value in args.concurrencies.split(",")]
    if len(set(concurrencies)) != len(concurrencies):
        parser.error("concurrency points must be unique")
    if args.cc_mode == "unknown" and not args.plan_only:
        parser.error("record the observed CC mode before running measurements")
    if args.natural_kind == "long-generation" and not args.dataset_path:
        parser.error("long-generation requires --dataset-path")
    minimum_requests = NATURAL_MIN_REQUESTS if args.dataset_path else MIN_REQUESTS
    requests_per_concurrency = NATURAL_REQUESTS_PER_CONCURRENCY if args.dataset_path else REQUESTS_PER_CONCURRENCY
    minimum_warmups = NATURAL_MIN_WARMUP_REQUESTS if args.dataset_path else MIN_WARMUP_REQUESTS
    warmups_per_concurrency = NATURAL_WARMUP_REQUESTS_PER_CONCURRENCY if args.dataset_path else WARMUP_REQUESTS_PER_CONCURRENCY
    expected_rows = None
    dataset_sha256 = None
    source_dataset_path = None
    if args.dataset_path:
        args.dataset_path = args.dataset_path.resolve()
        source_dataset_path = str(args.dataset_path)
        dataset_sha256 = hashlib.sha256(args.dataset_path.read_bytes()).hexdigest()
        dataset_rows = [json.loads(line) for line in args.dataset_path.read_text().splitlines() if line.strip()]
        needed = max(minimum_requests, requests_per_concurrency * max(concurrencies))
        required_keys = ("prompt", "case_id", "expected_json") if args.natural_kind == "oracle" else ("prompt", "case_id")
        if not dataset_rows or any(not all(key in row for key in required_keys) for row in dataset_rows):
            parser.error(f"natural corpus rows require {required_keys}")
        if args.natural_kind == "oracle":
            if len(dataset_rows) < needed:
                parser.error(f"oracle corpus needs at least {needed} rows")
            expected_rows = dataset_rows
    args.out_dir = args.out_dir.resolve()
    args.out_dir.mkdir(parents=True, exist_ok=False)
    if args.dataset_path and args.natural_kind == "long-generation":
        args.dataset_path = args.out_dir / "natural-expanded.jsonl"
        expanded = [{"case_id": f"{dataset_rows[index % len(dataset_rows)]['case_id']}-request-{index}", "prompt": f"{NATURAL_REQUEST_PREFIX} {index}.\n\n{dataset_rows[index % len(dataset_rows)]['prompt']}", "output_tokens": args.output_tokens} for index in range(needed)]
        args.dataset_path.write_text("".join(json.dumps(row) + "\n" for row in expanded))
    run_id = str(uuid.uuid4())
    trials = []
    for concurrency in concurrencies:
        for repetition in (0, *range(args.start_repetition, args.start_repetition + args.repetitions)):
            warmup = repetition == 0
            trials.append({
                "name": f"c{concurrency}-{'warmup' if warmup else f'r{repetition}'}",
                "concurrency": concurrency,
                "repetition": repetition,
                "warmup": warmup,
                "requests": max(minimum_warmups, warmups_per_concurrency * concurrency) if warmup else max(minimum_requests, requests_per_concurrency * concurrency),
                "seed": args.seed + concurrency * CONCURRENCY_SEED_STRIDE + repetition,
            })
    for trial in trials:
        trial["command"] = command_for(args, trial, run_id)
    manifest = {"created_at": datetime.now(timezone.utc).isoformat(), "run_id": run_id, "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}, "trials": trials}
    if args.dataset_path:
        manifest["source_dataset_path"] = source_dataset_path
        manifest["source_dataset_sha256"] = dataset_sha256
        manifest["client_dataset_sha256"] = hashlib.sha256(args.dataset_path.read_bytes()).hexdigest()
    save_json(args.out_dir / "manifest.json", manifest)
    if args.plan_only:
        return 0

    summary = {"status": "running", "trials": [], "aggregates": {}}
    try:
        versions = {}
        for package in CLIENT_PACKAGES:
            try:
                versions[package] = metadata.version(package)
            except metadata.PackageNotFoundError:
                versions[package] = None
        save_json(args.out_dir / "client-package-versions.json", versions)
        help_command = [sys.executable, "-m", "vllm.entrypoints.cli.main", "bench", "serve", "--help=all"]
        run_command(help_command, args.out_dir / "client-help.txt", remaining_seconds(deadline, HELP_TIMEOUT_SECONDS))
        help_text = (args.out_dir / "client-help.txt").read_text()
        flags = {argument for trial in trials for argument in trial["command"] if argument.startswith("--")}
        missing = sorted(flag for flag in flags if flag not in help_text)
        if missing:
            raise ValueError(f"installed benchmark client lacks expected flags: {missing}")
        for trial in trials:
            summary["active_trial"] = trial["name"]
            remaining = deadline - time.monotonic()
            if remaining < MIN_REMAINING_SECONDS:
                raise TimeoutError("sweep budget exhausted before the next trial")
            before = capture_metrics(args.base_url, args.out_dir / f"{trial['name']}.metrics-before.txt", remaining_seconds(deadline, METRICS_TIMEOUT_SECONDS))
            finish_counts(before)
            run_command(trial["command"], args.out_dir / f"{trial['name']}.log", remaining_seconds(deadline, args.trial_timeout_seconds))
            after = capture_metrics(args.base_url, args.out_dir / f"{trial['name']}.metrics-after.txt", remaining_seconds(deadline, METRICS_TIMEOUT_SECONDS))
            expected_reason = "stop" if args.dataset_path else "length"
            terminations = validate_finish_counts(before, after, trial["requests"], expected_reason)
            save_json(args.out_dir / f"{trial['name']}.finish-counts.json", {"expected_reason": expected_reason, "deltas": terminations})
            result_path = args.out_dir / f"{trial['name']}.json"
            result = json.loads(result_path.read_text())
            final_lens = None
            if args.dataset_path and args.natural_kind == "long-generation":
                counts_path = args.out_dir / f"{trial['name']}.final-token-lengths.json"
                count_command = [sys.executable, "-c", FINAL_TOKEN_COUNT_PROGRAM, args.tokenizer, str(result_path), str(counts_path)]
                run_command(count_command, args.out_dir / f"{trial['name']}.tokenizer.log", remaining_seconds(deadline, args.trial_timeout_seconds))
                final_lens = json.loads(counts_path.read_text())
            result.update(validate_result(result, trial["requests"], None if args.dataset_path else args.input_tokens, args.output_tokens, expected_rows, final_lens, args.min_final_tokens))
            remaining_seconds(deadline, args.trial_timeout_seconds)
            if final_lens is not None:
                save_json(args.out_dir / f"{trial['name']}.final-token-counts.json", {"final_output_lens": result["final_output_lens"], "final_output_throughput": result["final_output_throughput"]})
            record = {"name": trial["name"], "concurrency": trial["concurrency"], "warmup": trial["warmup"], "finish_counts": terminations, "metrics": {name: result.get(name) for name in METRIC_NAMES}}
            summary["trials"].append(record)
            save_json(args.out_dir / "summary.json", summary)
            sys.stdout.write(json.dumps(record) + "\n")
            sys.stdout.flush()
        for concurrency in concurrencies:
            measured = [trial for trial in summary["trials"] if trial["concurrency"] == concurrency and not trial["warmup"]]
            summary["aggregates"][str(concurrency)] = {name: {"median": statistics.median(values), "min": min(values), "max": max(values), "values": values} for name in METRIC_NAMES if (values := [trial["metrics"][name] for trial in measured if trial["metrics"][name] is not None])}
        summary["status"] = "complete"
        summary.pop("active_trial", None)
    except (Exception, KeyboardInterrupt) as exc:
        summary["status"] = "failed_or_incomplete"
        summary["error"] = f"{type(exc).__name__}: {exc}"
        sys.stderr.write(summary["error"] + "\n")
    save_json(args.out_dir / "summary.json", summary)
    return 0 if summary["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
