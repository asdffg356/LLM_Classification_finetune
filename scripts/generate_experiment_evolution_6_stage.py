"""Generate the six-stage experiment evolution figure.

The figure tracks validation log loss across six model iterations. Kaggle
Public Scores are displayed only as annotations so they are not confused with
validation log loss values.
"""

from pathlib import Path

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.image as mpimg
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    matplotlib = None
    mpimg = None
    plt = None



OUTPUT_PATH = Path("outputs/figures/experiment_evolution_6_stage.png")
MIN_FILE_SIZE_BYTES = 20 * 1024

STAGES = [
    "Stage 1\nInitial TF-IDF",
    "Stage 2\nTuned TF-IDF",
    "Stage 3\nDistilBERT Debug",
    "Stage 4\nTF-IDF + A/B Swap",
    "Stage 5\nWord+Char Ensemble",
    "Stage 6\nPairwise TF-IDF",
]
VALIDATION_LOG_LOSS = [
    1.139320,
    1.080877,
    1.078755,
    1.078106,
    1.077990,
    1.017894,
]
KAGGLE_PUBLIC_SCORES = {
    1: 1.07737,
    4: 1.07430,
    5: 1.02159,
}


def generate_fallback_png(output_path: Path = OUTPUT_PATH) -> Path:
    """Dependency-free fallback used only when matplotlib is unavailable."""
    import struct
    import zlib

    width, height = 2800, 1600
    pixels = bytearray([255, 255, 255] * width * height)

    def set_pixel(x: int, y: int, color: tuple[int, int, int]) -> None:
        if 0 <= x < width and 0 <= y < height:
            offset = (y * width + x) * 3
            pixels[offset : offset + 3] = bytes(color)

    def line(x0: int, y0: int, x1: int, y1: int, color: tuple[int, int, int], thick: int = 1) -> None:
        dx, dy = abs(x1 - x0), -abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx + dy
        while True:
            for tx in range(-thick, thick + 1):
                for ty in range(-thick, thick + 1):
                    set_pixel(x0 + tx, y0 + ty, color)
            if x0 == x1 and y0 == y1:
                break
            e2 = 2 * err
            if e2 >= dy:
                err += dy
                x0 += sx
            if e2 <= dx:
                err += dx
                y0 += sy

    def circle(cx: int, cy: int, radius: int, color: tuple[int, int, int]) -> None:
        for y in range(cy - radius, cy + radius + 1):
            for x in range(cx - radius, cx + radius + 1):
                if (x - cx) ** 2 + (y - cy) ** 2 <= radius ** 2:
                    set_pixel(x, y, color)

    left, right, top, bottom = 220, 130, 170, 320
    ymin, ymax = 1.005, 1.150

    def x_map(i: int) -> int:
        return int(left + i * (width - left - right) / (len(STAGES) - 1))

    def y_map(v: float) -> int:
        return int(height - bottom - (v - ymin) / (ymax - ymin) * (height - top - bottom))

    for tick in [1.02, 1.04, 1.06, 1.08, 1.10, 1.12, 1.14]:
        y = y_map(tick)
        line(left, y, width - right, y, (225, 231, 239), 1)
    line(left, top, left, height - bottom, (107, 114, 128), 2)
    line(left, height - bottom, width - right, height - bottom, (107, 114, 128), 2)

    points = [(x_map(i), y_map(v)) for i, v in enumerate(VALIDATION_LOG_LOSS)]
    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        line(x0, y0, x1, y1, (37, 99, 235), 5)
    for i, (x, y) in enumerate(points):
        circle(x, y, 22 if i < 5 else 38, (220, 38, 38) if i == 5 else (37, 99, 235))
        line(x, height - bottom - 15, x, height - bottom + 15, (107, 114, 128), 2)

    # Add colored annotation boxes where Kaggle Public Scores belong; the full
    # text labels are rendered by the matplotlib path in normal environments.
    for idx in KAGGLE_PUBLIC_SCORES:
        x, y = points[idx]
        for yy in range(y + 55, y + 125):
            for xx in range(x - 120, x + 120):
                set_pixel(xx, yy, (255, 247, 237))
        line(x - 120, y + 55, x + 120, y + 55, (251, 146, 60), 2)
        line(x - 120, y + 125, x + 120, y + 125, (251, 146, 60), 2)
        line(x - 120, y + 55, x - 120, y + 125, (251, 146, 60), 2)
        line(x + 120, y + 55, x + 120, y + 125, (251, 146, 60), 2)

    raw = b''.join(b'\x00' + pixels[y * width * 3 : (y + 1) * width * 3] for y in range(height))

    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack('>I', len(data)) + tag + data + struct.pack('>I', zlib.crc32(tag + data) & 0xFFFFFFFF)

    png = b'\x89PNG\r\n\x1a\n'
    png += chunk(b'IHDR', struct.pack('>IIBBBBB', width, height, 8, 2, 0, 0, 0))
    png += chunk(b'IDAT', zlib.compress(raw, 1))
    png += chunk(b'IEND', b'')
    output_path.write_bytes(png)

    file_size = output_path.stat().st_size
    if file_size < MIN_FILE_SIZE_BYTES:
        raise RuntimeError(f"Generated image is too small: {output_path} ({file_size} bytes)")
    if not output_path.read_bytes().startswith(b'\x89PNG\r\n\x1a\n'):
        raise RuntimeError(f"Generated image could not be read correctly: {output_path}")
    print("Matplotlib is not installed; used dependency-free PNG fallback.")
    return output_path


def generate_figure(output_path: Path = OUTPUT_PATH) -> Path:
    """Create and validate the experiment evolution PNG file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if plt is None:
        return generate_fallback_png(output_path)

    x_positions = list(range(1, len(STAGES) + 1))

    plt.style.use("default")
    fig, ax = plt.subplots(figsize=(14, 8), dpi=300)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#fbfdff")

    line_color = "#2563eb"
    best_color = "#dc2626"

    ax.plot(
        x_positions,
        VALIDATION_LOG_LOSS,
        color=line_color,
        linewidth=2.8,
        marker="o",
        markersize=8,
        markerfacecolor="white",
        markeredgecolor=line_color,
        markeredgewidth=2.2,
        label="Validation Log Loss",
        zorder=3,
    )

    best_index = len(STAGES) - 1
    ax.scatter(
        x_positions[best_index],
        VALIDATION_LOG_LOSS[best_index],
        s=220,
        color=best_color,
        edgecolor="white",
        linewidth=2.5,
        label="Best Stage",
        zorder=5,
    )

    for index, (x_value, y_value) in enumerate(zip(x_positions, VALIDATION_LOG_LOSS)):
        value_offset = 0.006 if index == best_index else 0.0045
        ax.annotate(
            f"{y_value:.6f}",
            xy=(x_value, y_value),
            xytext=(0, 15 if index != best_index else -24),
            textcoords="offset points",
            ha="center",
            va="bottom" if index != best_index else "top",
            fontsize=10.5,
            fontweight="bold" if index == best_index else "normal",
            color=best_color if index == best_index else "#1f2937",
            zorder=6,
        )

        if index in KAGGLE_PUBLIC_SCORES:
            kaggle_score = KAGGLE_PUBLIC_SCORES[index]
            y_offset = -50 if index != best_index else 34
            ax.annotate(
                f"Kaggle: {kaggle_score:.5f}",
                xy=(x_value, y_value),
                xytext=(0, y_offset),
                textcoords="offset points",
                ha="center",
                va="center",
                fontsize=10,
                color="#374151",
                bbox={
                    "boxstyle": "round,pad=0.35",
                    "facecolor": "#fff7ed",
                    "edgecolor": "#fb923c",
                    "linewidth": 1.1,
                    "alpha": 0.95,
                },
                arrowprops={
                    "arrowstyle": "->",
                    "color": "#fb923c",
                    "linewidth": 1.0,
                    "shrinkA": 4,
                    "shrinkB": 5,
                },
                zorder=7,
            )

    ax.annotate(
        "Final best\nPairwise TF-IDF",
        xy=(x_positions[best_index], VALIDATION_LOG_LOSS[best_index]),
        xytext=(-75, -5),
        textcoords="offset points",
        ha="right",
        va="center",
        fontsize=11,
        fontweight="bold",
        color=best_color,
        arrowprops={"arrowstyle": "->", "color": best_color, "linewidth": 1.5},
        zorder=7,
    )

    ax.set_title("Six-Stage Experiment Evolution", fontsize=20, fontweight="bold", pad=28)
    ax.text(
        0.5,
        1.025,
        "Validation Log Loss decreases across model iterations; Kaggle Public Score shown as annotations.",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=11.5,
        color="#4b5563",
    )

    ax.set_xlabel("Experiment Stage", fontsize=12.5, labelpad=16)
    ax.set_ylabel("Validation Log Loss", fontsize=12.5, labelpad=12)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(STAGES, fontsize=10.5)
    ax.set_xlim(0.65, len(STAGES) + 0.35)
    ax.set_ylim(1.005, 1.150)
    ax.grid(True, axis="y", color="#d1d5db", linestyle="--", linewidth=0.8, alpha=0.75)
    ax.grid(True, axis="x", color="#e5e7eb", linestyle=":", linewidth=0.7, alpha=0.55)

    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color("#9ca3af")
    ax.spines["bottom"].set_color("#9ca3af")
    ax.tick_params(axis="both", colors="#374151")
    ax.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="#d1d5db")

    fig.tight_layout(rect=(0.02, 0.03, 0.98, 0.93))
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    file_size = output_path.stat().st_size if output_path.exists() else 0
    if file_size < MIN_FILE_SIZE_BYTES:
        raise RuntimeError(
            f"Generated image is too small or missing: {output_path} ({file_size} bytes)"
        )

    image_data = mpimg.imread(output_path)
    if image_data.size == 0:
        raise RuntimeError(f"Generated image could not be read correctly: {output_path}")

    return output_path


if __name__ == "__main__":
    saved_path = generate_figure()
    print(f"Saved figure: {saved_path}")
    print(f"File size (bytes): {saved_path.stat().st_size}")
    print("PNG validation: readable")
