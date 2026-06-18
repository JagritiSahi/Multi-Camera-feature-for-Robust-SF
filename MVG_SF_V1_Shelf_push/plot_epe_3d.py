import re
import sys
import matplotlib.pyplot as plt
from pathlib import Path

def parse_epe_3d_per_step(log_path):
    """
    Parse 'Step X/100:' blocks and extract the EPE 3D value
    from the following 'EPE - ... 3D: <value>' line.
    """
    print(f"Parsing log file: {log_path}")
    step_pattern = re.compile(r"^Step\s+(\d+)/\d+:")
    epe_pattern = re.compile(r"EPE\s*-\s*.*3D:\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)")

    steps = []
    values = []

    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        print(f"Reading log file: {log_path}")
        lines = f.readlines()
        print(f"Lines===Done")

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        m_step = step_pattern.match(line)
        if m_step:
            step_num = int(m_step.group(1))

            # Look ahead a few lines for the EPE line
            j = i + 1
            while j < len(lines) and j <= i + 5:
                epe_line = lines[j].strip()
                m_epe = epe_pattern.search(epe_line)
                if m_epe:
                    epe_3d = float(m_epe.group(1))
                    steps.append(step_num)
                    values.append(epe_3d)
                    break
                j += 1
            i = j  # continue from where we searched up to
        else:
            i += 1

    return steps, values

def main():
    if len(sys.argv) < 2:
        print("Usage: python plot_epe_3d.py /path/to/Shelf_100.log")
        sys.exit(1)

    log_path = Path(sys.argv[1])
    if not log_path.is_file():
        print(f"Log file not found: {log_path}")
        sys.exit(1)

    steps, epe_3d_values = parse_epe_3d_per_step(log_path)

    if not steps:
        print("No step/EPE 3D values found in the log.")
        sys.exit(1)

    # Sort by step number (just in case)
    paired = sorted(zip(steps, epe_3d_values), key=lambda x: x[0])
    steps_sorted, values_sorted = zip(*paired)

    # Keep only every 10th step: 10, 20, 30, ...
    filtered = [(s, v) for s, v in zip(steps_sorted, values_sorted) if s % 10 == 0]
    if not filtered:
        print("No steps divisible by 10 were found; nothing to plot.")
        sys.exit(1)

    steps_filtered, values_filtered = zip(*filtered)

    print(f"Parsed {len(steps_sorted)} total steps, plotting {len(steps_filtered)} (every 10th).")
    print("Steps being plotted:", list(steps_filtered))

    plt.figure(figsize=(8, 5))
    plt.plot(steps_filtered, values_filtered, marker="o")
    plt.xlabel("Step")
    plt.ylabel("EPE 3D")
    plt.title(f"EPE 3D vs Step (every 10th) - {log_path.name}")
    plt.grid(True)
    plt.xticks(list(steps_filtered))
    plt.tight_layout()

    out_path = log_path.with_suffix(".epe3d_10.png")
    plt.savefig(out_path, dpi=150)
    print(f"Saved plot to: {out_path}")

    # If you have a GUI and want to see the window, uncomment:
    # plt.show()

if __name__ == "__main__":
    main()