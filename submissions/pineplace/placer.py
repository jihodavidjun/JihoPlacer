import os
import traceback

from pine_place.v1.current_engine import PinePlace as V1PinePlace
from pine_place.v1.current_engine import _load_plc_for_exact


class PinePlace(V1PinePlace):
    def place(self, benchmark):
        best_placement = super().place(benchmark)
        use_sa = os.environ.get("PINE_SA_POLISH", "0") == "1"
        use_v2 = os.environ.get("PINE_V2_REFINE", "0") == "1"
        use_v3 = os.environ.get("PINE_V3_GLOBAL", "0") == "1"
        plc = None
        compute_proxy_cost = None
        if use_sa or use_v2 or use_v3:
            try:
                from macro_place.objective import compute_proxy_cost as _compute_proxy_cost

                compute_proxy_cost = _compute_proxy_cost
                plc = _load_plc_for_exact(getattr(benchmark, "name", ""))
            except Exception as exc:
                print(f"[SA/V2/V3] skipped: exact setup unavailable: {exc}")
                plc = None

        if use_sa:
            if plc is None or compute_proxy_cost is None:
                print("[SA] skipped: exact PlacementCost unavailable")
            else:
                try:
                    from pine_place.v1.sa_polish import SAPolisher

                    time_budget = int(os.environ.get("PINE_SA_TIME", "180"))
                    polisher = SAPolisher(device="cuda")
                    sa_placement = polisher.polish(best_placement, benchmark, plc, time_budget_s=time_budget)
                    sa_cost = compute_proxy_cost(sa_placement, benchmark, plc)
                    current_cost = compute_proxy_cost(best_placement, benchmark, plc)
                    sa_proxy = float(sa_cost["proxy_cost"])
                    current_proxy = float(current_cost["proxy_cost"])
                    if sa_proxy < current_proxy and int(sa_cost.get("overlap_count", 0)) == 0:
                        print(f"[SA] improved: {current_proxy:.4f} -> {sa_proxy:.4f}")
                        best_placement = sa_placement
                    else:
                        print(f"[SA] no improvement: {sa_proxy:.4f}")
                except Exception as exc:
                    print(f"[SA] failed: {exc}")
                    traceback.print_exc()

        if use_v2:
            if plc is None or compute_proxy_cost is None:
                print("[V2] skipped: exact PlacementCost unavailable")
            else:
                try:
                    from pine_place.v2.refined_engine import V2RefinedEngine

                    iters = int(os.environ.get("PINE_V2_REFINE_ITERS", "1200"))
                    log_every = int(os.environ.get("PINE_V2_REFINE_LOG_EVERY", "50"))
                    max_nets_raw = os.environ.get("PINE_V2_REFINE_MAX_NETS", "")
                    max_nets = int(max_nets_raw) if max_nets_raw.strip() else None
                    engine = V2RefinedEngine(iterations=iters, log_every=log_every, max_nets=max_nets)
                    refined = engine.refine(best_placement, benchmark, plc)
                    refined_cost = compute_proxy_cost(refined, benchmark, plc)
                    v1_cost = compute_proxy_cost(best_placement, benchmark, plc)
                    refined_proxy = float(refined_cost["proxy_cost"])
                    v1_proxy = float(v1_cost["proxy_cost"])
                    if refined_proxy < v1_proxy and int(refined_cost.get("overlap_count", 0)) == 0:
                        print(f"[V2] improved: {v1_proxy:.4f} -> {refined_proxy:.4f}")
                        best_placement = refined
                    else:
                        print(f"[V2] no improvement: {refined_proxy:.4f} vs {v1_proxy:.4f}")
                except Exception as exc:
                    print(f"[V2] failed, using v1: {exc}")

        if use_v3:
            if plc is None:
                print("[V3] skipped: exact PlacementCost unavailable")
            else:
                try:
                    from pine_place.v3.global_engine import V3GlobalEngine

                    max_time = float(os.environ.get("PINE_V3_MAX_TIME_SECONDS", "300"))
                    starts = int(os.environ.get("PINE_V3_NUM_STARTS", "3"))
                    iters = int(os.environ.get("PINE_V3_ITERS", "1200"))
                    engine = V3GlobalEngine(device="cuda", max_time_seconds=max_time, num_starts=starts, iterations=iters)
                    v3_placement = engine.place(benchmark, plc)
                    v3_cost = compute_proxy_cost(v3_placement, benchmark, plc)
                    current_cost = compute_proxy_cost(best_placement, benchmark, plc)
                    v3_proxy = float(v3_cost["proxy_cost"])
                    current_proxy = float(current_cost["proxy_cost"])
                    if v3_proxy < current_proxy and int(v3_cost.get("overlap_count", 0)) == 0:
                        print(f"[V3] improved: {current_proxy:.4f} -> {v3_proxy:.4f}")
                        best_placement = v3_placement
                    else:
                        print(f"[V3] no improvement: {v3_proxy:.4f} vs {current_proxy:.4f}")
                except Exception as exc:
                    print(f"[V3] failed, using v1: {exc}")
                    traceback.print_exc()
        return best_placement
