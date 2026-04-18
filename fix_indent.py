import sys

with open("streamlit_app.py", "r") as f:
    lines = f.readlines()

new_lines = []
for i, line in enumerate(lines):
    if i >= 1079 and i <= 1490: # Covers the entire injected block + original following block properly
        if line.startswith("            "):
            new_lines.append(line[8:])
        elif line.startswith("        "):
            new_lines.append(line[4:])
        elif line.strip() == "":
            new_lines.append("\n")
        else:
            new_lines.append(line)
    else:
        new_lines.append(line)

with open("streamlit_app.py", "w") as f:
    f.writelines(new_lines)
