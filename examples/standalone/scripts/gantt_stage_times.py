import sys
import pandas as pd

assert len(sys.argv) in [2, 3]

csv_file = sys.argv[1]
output_file = None if len(sys.argv) == 2 else sys.argv[2]

# Load CSV
df = pd.read_csv(csv_file)

# Pivot to get start and end times
# Assuming consecutive rows are start and end of each stage
gantt = []
gantt.append("  section Run 0")
stages = df["stage"].tolist()
times = df["time_ms"].tolist()
current = 0
for i in range(0, len(stages)):
    stage_name = stages[i]
    start = current
    end = start + times[i]
    current = end
    gantt.append(f"    {stage_name} Stage : {start}, {end}")

content = ""
content += "```mermaid\n"
content += "gantt\n"
content += "  title Flow\n"
content += "  dateFormat x\n"
content += "  axisFormat %H:%M:%S\n"
content += "\n".join(gantt)
content += "\n```"

# Write Mermaid Gantt
if output_file is None:
    print(content)
else:
    with open(output_file, "w") as f:
        f.write(content)
    print(f"Mermaid Gantt chart written to {output_file}")
