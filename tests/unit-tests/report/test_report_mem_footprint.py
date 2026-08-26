import pandas as pd

from isaac_toolkit.report.report_mem_footprint import (
    classify_section,
    generate_code_size_summary,
    generate_function_table,
)


def test_code_size_summary_calculates_rom_and_ram():
    sections = pd.DataFrame(
        {
            "name": [".text", ".text.hot", ".rodata", ".data", ".bss", ".debug_info"],
            "data_size": [100, 20, 30, 10, 40, 1000],
        }
    )
    functions = pd.DataFrame({"func": ["foo", "bar"], "bytes": [70, 40]})

    summary = generate_code_size_summary(sections, functions).set_index("Metric")["Bytes"]

    assert summary["Code"] == 120
    assert summary["Estimated ROM"] == 160
    assert summary["Estimated RAM"] == 50
    assert summary["Function symbols"] == 110


def test_non_allocated_and_array_sections_are_not_misclassified():
    assert classify_section(".debug_info") == "Other"
    assert classify_section(".init_array") == "Other"
    assert classify_section(".srodata.str1.4") == "Read-only data"


def test_function_table_derives_share_when_missing():
    functions = pd.DataFrame({"func": ["small", "large"], "bytes": [25, 75]})

    result = generate_function_table(functions)

    assert result["Function"].tolist() == ["large", "small"]
    assert result["Share"].tolist() == [0.75, 0.25]
