#!/usr/bin/env python3
"""
benchmark_blockchain_overhead.py

Measures blockchain (gas, smart-contract execution time) and IPFS (upload/
download latency) overhead for the DCMF-BFL system.

Designed to survive SSH disconnects:
  * Logs to BOTH stdout and a timestamped log file under --output_dir
  * Writes a checkpoint after every round; on restart, resumes from where it
    stopped (no double-counting, no wasted gas)
  * Catches SIGINT / SIGTERM and saves a final checkpoint before exiting
  * Reads contract address from config.yaml by default; --addr overrides

Typical usage (inside a tmux window):
    python benchmark_blockchain_overhead.py \
        --rounds 10 --clients 5 --model_type cnn \
        --ipfs_reps 20 --output_dir benchmark_results

Resume after an interruption: just run the SAME command again. The script
will detect the checkpoint and pick up at the next unfinished round.

To force a clean run: delete --output_dir or pass --fresh.
"""

import argparse
import json
import logging
import os
import signal
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
import yaml

from blockchain_utils import BlockchainUtils
from ipfs_utils import IPFSUtils
from model import get_model

# ---------------------------------------------------------------------------
# Defaults (overridable via config.yaml or CLI)
# ---------------------------------------------------------------------------
DEFAULT_URL = "http://127.0.0.1:7545"
DEFAULT_ABI_PATH = "build/contracts/BCFL.json"
# Set this to your most recent deployed address as a fallback. config.yaml
# wins over this; --addr wins over both.
DEFAULT_ADDR = "0xe78A0F7E598Cc8b0Bb87894B0F60dD2a88d6a8Ab"

# Pricing assumptions for the "estimated mainnet cost" column
DEFAULT_GAS_PRICE_GWEI = 30
DEFAULT_ETH_USD = 2500


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_config(config_path):
    if config_path and os.path.exists(config_path):
        with open(config_path, "r") as f:
            return yaml.safe_load(f) or {}
    return {}


def load_abi(abi_path):
    with open(abi_path, "r") as f:
        return json.load(f)["abi"]


def setup_logging(output_dir):
    """Send every log message to BOTH stdout and a file in output_dir."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    log_path = os.path.join(
        output_dir, f"benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    )
    fmt = "%(asctime)s - %(levelname)s - %(message)s"
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    # Reset any previously configured handlers (prevents double-printing on resume)
    root.handlers.clear()
    fh = logging.FileHandler(log_path)
    fh.setFormatter(logging.Formatter(fmt))
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(logging.Formatter(fmt))
    root.addHandler(fh)
    root.addHandler(sh)
    logging.info(f"Log file: {log_path}")
    return log_path


def checkpoint_path(output_dir):
    return os.path.join(output_dir, "checkpoint.json")


def save_checkpoint(output_dir, results):
    tmp = checkpoint_path(output_dir) + ".tmp"
    with open(tmp, "w") as f:
        json.dump(results, f, indent=2)
    os.replace(tmp, checkpoint_path(output_dir))  # atomic on POSIX


def load_checkpoint(output_dir):
    p = checkpoint_path(output_dir)
    if not os.path.exists(p):
        return None
    with open(p, "r") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Measurement primitives
# ---------------------------------------------------------------------------
def measure_ipfs_latency(ipfs, model, reps):
    """Upload + download the model `reps` times, return latency stats in ms."""
    upload_ms, download_ms, sample_cids = [], [], []
    logging.info(f"Measuring IPFS latency over {reps} reps ...")
    for i in range(reps):
        t0 = time.perf_counter()
        cid = ipfs.upload_model(model)
        upload_ms.append((time.perf_counter() - t0) * 1000)
        if cid is None:
            raise RuntimeError(f"IPFS upload returned None on rep {i}")
        sample_cids.append(cid)

        t0 = time.perf_counter()
        _ = ipfs.download_model(cid)
        download_ms.append((time.perf_counter() - t0) * 1000)

        if (i + 1) % max(1, reps // 4) == 0:
            logging.info(f"  IPFS rep {i + 1}/{reps}")

    def stats(xs):
        return {
            "mean_ms": statistics.mean(xs),
            "std_ms": statistics.stdev(xs) if len(xs) > 1 else 0.0,
            "min_ms": min(xs),
            "max_ms": max(xs),
            "all_ms": xs,
        }

    return {
        "reps": reps,
        "upload": stats(upload_ms),
        "download": stats(download_ms),
        "sample_cids": sample_cids[:3],
    }


def gas_and_time(web3, tx_hash, t_start):
    """Wait for a tx receipt, return (gasUsed, wall_time_ms)."""
    receipt = web3.eth.wait_for_transaction_receipt(tx_hash)
    return receipt.gasUsed, (time.perf_counter() - t_start) * 1000


def run_single_round(blockchain, ipfs, model, owner, trainer_addrs, current_round):
    """Execute one full FL round on-chain and return per-call gas/time."""
    web3 = blockchain.web3
    contract = blockchain.contract
    round_data = {"round": current_round, "calls": {}}
    round_t0 = time.perf_counter()

    # 1) selectTrainersForRound
    t0 = time.perf_counter()
    tx = contract.functions.selectTrainersForRound(
        current_round, list(trainer_addrs)
    ).transact({"from": owner})
    gas, ms = gas_and_time(web3, tx, t0)
    round_data["calls"]["selectTrainersForRound"] = {"gas": gas, "time_ms": ms}
    logging.info(f"  selectTrainersForRound  gas={gas:>8}  t={ms:.1f} ms")

    # 2) submitUpdate (one per client)
    submit_update = []
    for ci, addr in enumerate(trainer_addrs):
        cid = ipfs.upload_model(model)
        if cid is None:
            raise RuntimeError("IPFS upload failed during submitUpdate")
        t0 = time.perf_counter()
        tx = contract.functions.submitUpdate(current_round, cid).transact({"from": addr})
        gas, ms = gas_and_time(web3, tx, t0)
        submit_update.append({"client": ci, "gas": gas, "time_ms": ms, "cid": cid})
    round_data["calls"]["submitUpdate"] = submit_update
    avg = sum(d["gas"] for d in submit_update) / len(submit_update)
    logging.info(f"  submitUpdate  n={len(submit_update)}  avg_gas={avg:,.0f}")

    # 3) submitScore (one per client; owner is also evaluator by default)
    submit_score = []
    for ci, addr in enumerate(trainer_addrs):
        score = 100 - ci  # dummy non-zero score so distributeTokens works
        t0 = time.perf_counter()
        tx = contract.functions.submitScore(current_round, addr, score).transact(
            {"from": owner}
        )
        gas, ms = gas_and_time(web3, tx, t0)
        submit_score.append({"client": ci, "gas": gas, "time_ms": ms})
    round_data["calls"]["submitScore"] = submit_score
    avg = sum(d["gas"] for d in submit_score) / len(submit_score)
    logging.info(f"  submitScore   n={len(submit_score)}  avg_gas={avg:,.0f}")

    # 4) submitGlobalModel (advances currentRound)
    global_cid = ipfs.upload_model(model)
    if global_cid is None:
        raise RuntimeError("IPFS upload failed during submitGlobalModel")
    t0 = time.perf_counter()
    tx = contract.functions.submitGlobalModel(current_round, global_cid).transact(
        {"from": owner}
    )
    gas, ms = gas_and_time(web3, tx, t0)
    round_data["calls"]["submitGlobalModel"] = {"gas": gas, "time_ms": ms, "cid": global_cid}
    logging.info(f"  submitGlobalModel  gas={gas:>8}  t={ms:.1f} ms")

    # 5) distributeTokens (round just completed; round < currentRound now)
    try:
        t0 = time.perf_counter()
        tx = contract.functions.distributeTokens(current_round, 1_000_000).transact(
            {"from": owner}
        )
        gas, ms = gas_and_time(web3, tx, t0)
        round_data["calls"]["distributeTokens"] = {"gas": gas, "time_ms": ms}
        logging.info(f"  distributeTokens   gas={gas:>8}  t={ms:.1f} ms")
    except Exception as e:
        logging.warning(f"  distributeTokens failed: {e}")
        round_data["calls"]["distributeTokens"] = {"gas": 0, "time_ms": 0, "error": str(e)}

    round_data["total_sc_time_ms"] = (time.perf_counter() - round_t0) * 1000
    round_data["finished_at"] = time.time()
    return round_data


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
def compute_summary(results, n_clients, gas_price_gwei, eth_usd):
    rounds = results.get("rounds", [])
    if not rounds:
        return {}

    summary = {"per_function_gas": {}}

    # Single-call functions
    for fn in ("selectTrainersForRound", "submitGlobalModel"):
        gases = [r["calls"][fn]["gas"] for r in rounds if fn in r["calls"]]
        if gases:
            summary["per_function_gas"][fn] = {
                "mean": statistics.mean(gases),
                "std": statistics.stdev(gases) if len(gases) > 1 else 0,
                "n": len(gases),
            }

    # Per-client functions
    for fn in ("submitUpdate", "submitScore"):
        gases = [c["gas"] for r in rounds for c in r["calls"].get(fn, [])]
        if gases:
            summary["per_function_gas"][fn] = {
                "mean": statistics.mean(gases),
                "std": statistics.stdev(gases) if len(gases) > 1 else 0,
                "n": len(gases),
            }

    # distributeTokens (skip errored ones)
    dt = [
        r["calls"]["distributeTokens"]["gas"]
        for r in rounds
        if "distributeTokens" in r["calls"]
        and "error" not in r["calls"]["distributeTokens"]
    ]
    if dt:
        summary["per_function_gas"]["distributeTokens"] = {
            "mean": statistics.mean(dt),
            "std": statistics.stdev(dt) if len(dt) > 1 else 0,
            "n": len(dt),
        }

    # Total per-round gas
    per_round_total = []
    for r in rounds:
        total = 0
        for v in r["calls"].values():
            if isinstance(v, list):
                total += sum(d["gas"] for d in v)
            elif "gas" in v:
                total += v["gas"]
        per_round_total.append(total)
    summary["per_round_gas"] = {
        "mean": statistics.mean(per_round_total),
        "std": statistics.stdev(per_round_total) if len(per_round_total) > 1 else 0,
    }

    # Per-round SC wall time
    sc_times = [r["total_sc_time_ms"] for r in rounds]
    summary["per_round_sc_time_ms"] = {
        "mean": statistics.mean(sc_times),
        "std": statistics.stdev(sc_times) if len(sc_times) > 1 else 0,
    }

    # Estimated cost @ given gas price
    gas_price_eth = gas_price_gwei * 1e-9
    cost_eth = summary["per_round_gas"]["mean"] * gas_price_eth
    summary["estimated_cost_per_round"] = {
        "gas_price_gwei": gas_price_gwei,
        "eth_usd": eth_usd,
        "eth": cost_eth,
        "usd": cost_eth * eth_usd,
    }
    summary["n_clients"] = n_clients
    return summary


def print_summary(results):
    s = results.get("summary", {})
    bar = "=" * 64
    print(f"\n{bar}\nBENCHMARK SUMMARY\n{bar}")
    print("\nPer-function gas (mean ± std):")
    for fn, d in s.get("per_function_gas", {}).items():
        print(f"  {fn:<28s}{d['mean']:>12,.0f} ± {d['std']:>10,.0f}  (n={d['n']})")
    if "per_round_gas" in s:
        print(
            f"\nTotal gas / round:   "
            f"{s['per_round_gas']['mean']:,.0f} ± {s['per_round_gas']['std']:,.0f}"
        )
    if "per_round_sc_time_ms" in s:
        print(
            f"SC time / round:     "
            f"{s['per_round_sc_time_ms']['mean']:.1f} ± "
            f"{s['per_round_sc_time_ms']['std']:.1f} ms"
        )
    if "estimated_cost_per_round" in s:
        c = s["estimated_cost_per_round"]
        print(
            f"\nEstimated mainnet cost / round: "
            f"{c['eth']:.6f} ETH  (~${c['usd']:.2f})"
        )
        print(f"  assumptions: {c['gas_price_gwei']} Gwei, ETH=${c['eth_usd']}")
    ipfs = results.get("ipfs_latency", {})
    if ipfs:
        print(
            f"\nIPFS upload:   "
            f"{ipfs['upload']['mean_ms']:.2f} ± {ipfs['upload']['std_ms']:.2f} ms "
            f"(n={ipfs['reps']})"
        )
        print(
            f"IPFS download: "
            f"{ipfs['download']['mean_ms']:.2f} ± {ipfs['download']['std_ms']:.2f} ms "
            f"(n={ipfs['reps']})"
        )
    print(bar)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def run_benchmark(args):
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    if args.fresh:
        cp = checkpoint_path(args.output_dir)
        if os.path.exists(cp):
            os.remove(cp)
            print(f"[fresh] removed {cp}")

    setup_logging(args.output_dir)

    cfg = load_config(args.config)
    contract_address = (
        args.addr
        or cfg.get("blockchain", {}).get("contract_address")
        or DEFAULT_ADDR
    )
    url = args.url or cfg.get("blockchain", {}).get("url") or DEFAULT_URL
    abi_path = args.abi_path or cfg.get("blockchain", {}).get("abi_path") or DEFAULT_ABI_PATH

    logging.info("=== Configuration ===")
    logging.info(f"  Web3 URL:    {url}")
    logging.info(f"  Contract:    {contract_address}")
    logging.info(f"  ABI path:    {abi_path}")
    logging.info(f"  Rounds:      {args.rounds}")
    logging.info(f"  Clients:     {args.clients}")
    logging.info(f"  Model type:  {args.model_type}")
    logging.info(f"  IPFS reps:   {args.ipfs_reps}")
    logging.info(f"  Output dir:  {args.output_dir}")

    abi = load_abi(abi_path)
    blockchain = BlockchainUtils(url, contract_address, abi)
    web3 = blockchain.web3
    if not web3.is_connected():
        logging.error("Web3 cannot connect to Ganache. Is ganache-cli running?")
        sys.exit(2)
    ipfs = IPFSUtils()

    # Resume from checkpoint if present
    results = load_checkpoint(args.output_dir)
    if results:
        logging.info(
            f"Resuming from checkpoint with {len(results.get('rounds', []))} "
            f"completed rounds."
        )
    else:
        results = {
            "config": vars(args),
            "contract_address": contract_address,
            "started_at": datetime.now().isoformat(),
            "rounds": [],
            "ipfs_latency": None,
            "init_gas": None,
        }

    # Save-on-signal so Ctrl-C / SIGTERM doesn't lose progress
    def _save_and_exit(signum, frame):
        logging.warning(f"Received signal {signum}. Saving checkpoint and exiting.")
        save_checkpoint(args.output_dir, results)
        sys.exit(130)

    signal.signal(signal.SIGINT, _save_and_exit)
    signal.signal(signal.SIGTERM, _save_and_exit)

    # Build the model used for IPFS roundtrips
    model = get_model(args.model_type)
    n_params = sum(p.numel() for p in model.parameters())
    size_kb = n_params * 4 / 1024  # float32 ≈ 4 bytes/param
    logging.info(
        f"Model: {args.model_type}  params={n_params:,}  ~{size_kb:.1f} KB"
    )
    results.setdefault("model_info", {
        "type": args.model_type,
        "n_params": n_params,
        "size_kb": size_kb,
    })

    # IPFS latency: do once, persist to checkpoint
    if results.get("ipfs_latency") is None:
        results["ipfs_latency"] = measure_ipfs_latency(ipfs, model, args.ipfs_reps)
        save_checkpoint(args.output_dir, results)

    # Get accounts
    accounts = web3.eth.accounts
    if len(accounts) < args.clients + 1:
        logging.error(
            f"Ganache only exposes {len(accounts)} accounts, but we need "
            f"{args.clients + 1} (1 owner + {args.clients} clients). "
            f"Restart ganache with `-a {args.clients + 1}` or larger."
        )
        sys.exit(3)
    owner = accounts[0]
    trainer_addrs = accounts[1: 1 + args.clients]

    # Initialize task if not yet done. Also measures the initialize() gas once.
    is_initialized = blockchain.contract.functions.task().call()[4]
    if not is_initialized and results.get("init_gas") is None:
        logging.info("Initializing task on-chain ...")
        genesis_cid = ipfs.upload_model(model)
        t0 = time.perf_counter()
        tx = blockchain.contract.functions.initialize(
            genesis_cid, args.rounds + 1, args.clients
        ).transact({"from": owner})
        gas, ms = gas_and_time(web3, tx, t0)
        results["init_gas"] = {"gas": gas, "time_ms": ms, "genesis_cid": genesis_cid}
        logging.info(f"  initialize  gas={gas:,}  t={ms:.1f} ms")

        # Round 0 is genesis — advance to round 1 by submitting the genesis CID
        t0 = time.perf_counter()
        tx = blockchain.contract.functions.submitGlobalModel(0, genesis_cid).transact(
            {"from": owner}
        )
        web3.eth.wait_for_transaction_receipt(tx)
        save_checkpoint(args.output_dir, results)
    elif is_initialized and results.get("init_gas") is None:
        # We can't measure initialize() retroactively (it can only run once).
        results["init_gas"] = {
            "gas": None,
            "time_ms": None,
            "note": "task was already initialized before this script started",
        }
        save_checkpoint(args.output_dir, results)

    # Main round loop
    completed = len(results["rounds"])
    for _ in range(completed, args.rounds):
        try:
            current_round = blockchain.contract.functions.getCurrentRound().call()
        except Exception as e:
            logging.error(f"Lost web3 connection: {e}. Saving checkpoint and exiting.")
            save_checkpoint(args.output_dir, results)
            sys.exit(4)

        logging.info(
            f"\n=== Round {len(results['rounds']) + 1}/{args.rounds}  "
            f"(on-chain round = {current_round}) ==="
        )
        round_data = run_single_round(
            blockchain, ipfs, model, owner, trainer_addrs, current_round
        )
        results["rounds"].append(round_data)
        save_checkpoint(args.output_dir, results)
        logging.info(
            f"  total SC wall time = {round_data['total_sc_time_ms']:.1f} ms  "
            f"[checkpoint saved]"
        )

    # Final summary
    results["summary"] = compute_summary(
        results, args.clients, args.gas_price_gwei, args.eth_usd
    )
    results["finished_at"] = datetime.now().isoformat()
    final_path = os.path.join(args.output_dir, "results.json")
    with open(final_path, "w") as f:
        json.dump(results, f, indent=2)
    logging.info(f"\nFinal results: {final_path}")
    print_summary(results)


def main():
    p = argparse.ArgumentParser(
        description="Benchmark blockchain + IPFS overhead for DCMF-BFL",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--rounds", type=int, default=10)
    p.add_argument("--clients", type=int, default=5)
    p.add_argument("--model_type", choices=["cnn", "resnet"], default="cnn")
    p.add_argument("--ipfs_reps", type=int, default=20)
    p.add_argument("--output_dir", default="benchmark_results")

    p.add_argument("--config", default="config.yaml",
                   help="Path to config.yaml. Set to '' to disable.")
    p.add_argument("--addr", default=None,
                   help="Contract address. Overrides config.yaml.")
    p.add_argument("--url", default=None,
                   help="Web3 URL. Overrides config.yaml.")
    p.add_argument("--abi_path", default=None,
                   help="Path to BCFL.json. Overrides config.yaml.")

    p.add_argument("--gas_price_gwei", type=float, default=DEFAULT_GAS_PRICE_GWEI)
    p.add_argument("--eth_usd", type=float, default=DEFAULT_ETH_USD)

    p.add_argument("--fresh", action="store_true",
                   help="Delete any existing checkpoint and start over.")

    args = p.parse_args()
    run_benchmark(args)


if __name__ == "__main__":
    main()