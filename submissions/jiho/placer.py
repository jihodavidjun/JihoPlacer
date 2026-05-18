import os

from jiho_place.v1.current_engine import JihoPlacer as V1JihoPlacer
from jiho_place.v1.current_engine import _load_plc_for_exact


class JihoPlacer(V1JihoPlacer):
    def place(self, benchmark):
        best_placement = super().place(benchmark)
        if os.environ.get("JIHO_V2_REFINE", "1") != "1":
            return best_placement

        try:
            from macro_place.objective import compute_proxy_cost

            plc = _load_plc_for_exact(getattr(benchmark, "name", ""))
            if plc is None:
                print("[V2] skipped: exact PlacementCost unavailable")
                return best_placement

            from jiho_place.v2.refined_engine import V2RefinedEngine

            iters = int(os.environ.get("JIHO_V2_REFINE_ITERS", "1200"))
            log_every = int(os.environ.get("JIHO_V2_REFINE_LOG_EVERY", "50"))
            max_nets_raw = os.environ.get("JIHO_V2_REFINE_MAX_NETS", "")
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
        return best_placement
