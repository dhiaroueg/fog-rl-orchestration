#!/usr/bin/env python3
"""Exporte MobileNetV2 au format ONNX pour la requete lourde du worker (WORKER_MODE=real, CDC section 6.1).

    pip install torchvision onnxruntime        # hors requirements : necessaire seulement pour cette etape
    python scripts/export_mobilenet_onnx.py                 # models_onnx/mobilenet_v2.onnx, poids aleatoires
    python scripts/export_mobilenet_onnx.py --pretrained    # poids ImageNet (telechargement par torchvision)

Pour la calibration et le routage, seul le COUT DE CALCUL compte : un MobileNetV2 a poids aleatoires a exactement le meme
cout que le reseau entraine. L'option --pretrained n'est utile que pour une demonstration de classification.
Le worker alimente le modele avec un tenseur float32 de forme (1, 3, 224, 224) (fogrl/worker/workloads.py) ;
le dossier de sortie est monte en lecture seule dans les conteneurs (deploy/compose.*.yml : ../models_onnx:/models:ro).

NOTE : script NON execute dans l'environnement de developpement du depot (torchvision y est absent, et le telechargement
des poids est hors de son acces reseau). Il se limite a l'API publique de torchvision et d'onnxruntime ; verifiez la duree
affichee en fin d'execution : pour une requete lourde, le CDC attend ~150 ms sur un coeur de vitesse 1.0.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

INPUT_SHAPE = (1, 3, 224, 224)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default="models_onnx/mobilenet_v2.onnx")
    parser.add_argument("--pretrained", action="store_true", help="poids ImageNet au lieu de poids aleatoires")
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--runs", type=int, default=20, help="inferences de verification (onnxruntime, 1 thread)")
    args = parser.parse_args(argv)

    try:
        import torch
        import torchvision
    except ImportError as exc:
        print(f"Dependance manquante ({exc}). Installez-la : pip install torchvision", file=sys.stderr)
        return 2

    weights = torchvision.models.MobileNet_V2_Weights.DEFAULT if args.pretrained else None
    model = torchvision.models.mobilenet_v2(weights=weights).eval()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        torch.randn(*INPUT_SHAPE),
        str(out),
        input_names=["input"],
        output_names=["output"],
        opset_version=args.opset,
    )
    print(f"Modele ecrit : {out} ({out.stat().st_size / 1e6:.1f} Mo)")

    try:
        import numpy as np
        import onnxruntime as ort
    except ImportError:
        print("onnxruntime absent : verification sautee (pip install onnxruntime).")
        return 0
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1  # comme le worker : un serveur = un coeur
    session = ort.InferenceSession(str(out), sess_options=options, providers=["CPUExecutionProvider"])
    name = session.get_inputs()[0].name
    sample = np.random.default_rng(0).random(INPUT_SHAPE, dtype=np.float32)
    session.run(None, {name: sample})  # echauffement
    durations = []
    for _ in range(args.runs):
        start = time.perf_counter()
        session.run(None, {name: sample})
        durations.append((time.perf_counter() - start) * 1000.0)
    print(f"Inference sur 1 coeur : mediane {statistics.median(durations):.1f} ms sur {args.runs} essais (CDC : ~150 ms)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
