import argparse
import json

from quantlab.cli import tensor_demo


def test_tensor_demo_emits_machine_readable_report(capsys) -> None:
    args = argparse.Namespace(seed=1, shape=[4, 8], scheme="symmetric", axis=0)

    tensor_demo(args)

    report = json.loads(capsys.readouterr().out)
    assert report["scheme"] == "symmetric"
    assert report["shape"] == [4, 8]
    assert report["compression_ratio_vs_fp32"] > 1
