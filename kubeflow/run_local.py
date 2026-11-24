"""Execute the Kubeflow pipeline locally with the KFP local runner (Docker)."""

import sys

from kfp import local

from kubeflow.btc_kfp_pipeline import btc_forecast_pipeline


def main(runner: str = "docker") -> None:
    chosen = local.DockerRunner() if runner == "docker" else local.SubprocessRunner(use_venv=True)
    local.init(runner=chosen, pipeline_root="./kfp_local_outputs")
    task = btc_forecast_pipeline(period="5y")
    print("Pipeline outputs:", task.outputs)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "docker")
