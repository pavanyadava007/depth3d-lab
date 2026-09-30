"""Download the public model weights used in this study into the Hugging Face cache.

Weights are never committed. Run once with SSL_CERT_FILE set if behind a proxy.
"""

from __future__ import annotations

import time

from huggingface_hub import hf_hub_download, snapshot_download


def main() -> None:
    t = time.time()
    p = hf_hub_download("apple/DepthPro", "depth_pro.pt")
    print("depthpro", p, round(time.time() - t, 1), flush=True)
    p = snapshot_download("depth-anything/Depth-Anything-V2-Metric-Outdoor-Large-hf")
    print("dav2", p, round(time.time() - t, 1), flush=True)
    p = hf_hub_download("facebook/VGGT-1B", "model.pt")
    print("vggt", p, round(time.time() - t, 1), flush=True)
    print("DOWNLOAD_DONE")


if __name__ == "__main__":
    main()
