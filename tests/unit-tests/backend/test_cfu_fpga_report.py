from pathlib import Path
import runpy

import pytest


def test_fpga_report_keeps_routed_timing_and_resource_stages(tmp_path):
    script = Path(__file__).resolve().parents[4] / "scripts/cfu_fpga_report.py"
    if not script.exists():
        pytest.skip("Demo report adapter requires the parent monorepo")
    extract = runpy.run_path(str(script))["extract"]
    gateware = tmp_path / "gateware"
    gateware.mkdir()
    (gateware / "digilent_arty_utilization_place.rpt").write_text(
        "| Slice LUTs | 2670 | 0 |\n| Slice Registers | 2159 | 0 |\n"
        "| Block RAM Tile | 40 | 0 |\n| DSPs | 4 | 0 |\n")
    timing = gateware / "digilent_arty_timing.rpt"
    timing.write_text("WNS(ns) TNS(ns) Endpoints Total WHS(ns)\n-------\n"
                      "1.246 0.000 0 5915 0.023\n"
                      "All user specified timing constraints are met.\n"
                      "main_crg_clkout0 {0.000 10.000} 20.000 50.000\n")
    (gateware / "digilent_arty_utilization_hierarchical_place.rpt").write_text(
        "| Cfu | Cfu | 1 | 1 | 0 | 0 | 33 | 0 | 0 | 0 |\n")
    result = extract(tmp_path)
    assert result["luts"] == 2670 and result["flip_flops"] == 2159
    assert result["wns_ns"] == 1.246 and result["timing_met"]
    assert result["constrained_frequency_mhz"] == 50
    assert result["resource_stage"] == "post-placement"
    assert result["timing_stage"] == "post-route"
    timing.write_text("Missing timing summary\n")
    with pytest.raises(ValueError, match="Missing routed timing"):
        extract(tmp_path)
