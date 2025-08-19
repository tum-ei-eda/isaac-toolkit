#
# Copyright (c) 2025 TUM Department of Electrical and Computer Engineering.
#
# This file is part of ISAAC Toolkit.
# See https://github.com/tum-ei-eda/isaac-toolkit.git for further info.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""Utilities to generate Reports."""
import base64
from pathlib import Path

from weasyprint import HTML


def save_pdf_report(html_str, out_path):
    """Convert HTML string into PDF file."""
    HTML(string=html_str).write_pdf(out_path)


JUPYTER_CSS = """
<style>
table.dataframe {
  border-collapse: collapse;
  border: 1px solid #aaa;
  font-family: sans-serif;
  margin-bottom: 1em;
}
table.dataframe th, table.dataframe td {
  border: 1px solid #aaa;
  padding: 4px 8px;
  text-align: right;
}
table.dataframe th {
  background-color: #f2f2f2;
}
table.dataframe tr:nth-child(even) {
  background-color: #fafafa;
}
table.dataframe tr:hover {
  background-color: #f5f5f5;
}

body {
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue",
               Arial, sans-serif, "Apple Color Emoji", "Segoe UI Emoji";
  margin: 20px;
  color: #333;
  line-height: 1.5;
}

h1, h2 {
  font-weight: 600;
  font-family: inherit;
  border-bottom: 2px solid #eaeaea;
  padding-bottom: 4px;
  margin-top: 1.5em;
  margin-bottom: 0.75em;
}

h1 {
  font-size: 1.8em;
  color: #222;
}

h2 {
  font-size: 1.4em;
  color: #444;
}

table.dataframe {
  margin-top: 1em;
}
</style>
"""


def encode_image_base64(path: Path) -> str:
    """Convert image into bas64 for HTML embedding."""
    with open(path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode("utf-8")


def format_hex(x):
    """Convert addresses into hexadecimal."""
    return hex(int(x))


def monospace_addresses_html(val):
    """Wrap hex addresses into HTML code tags."""
    if isinstance(val, str) and val.startswith("0x"):
        return f"<code>{val}</code>"
    return val


def monospace_addresses_md(val):
    """Wrap hex addresses into Markdown code tags."""
    if isinstance(val, str) and val.startswith("0x"):
        return f"`{val}`"
    return val
