
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Ellipse, Circle
from pathlib import Path

# ============================================================
# BEE POLLINATION + PARTICLE FILTER SIMULATION
# ============================================================

rng = np.random.default_rng(12)

DT = 0.12
FRAMES = 240
FPS = 20

N_FILTER = 450
N_POLLEN = 250

W, H = 20.0, 12.0

flowers = np.array([
    [3.0, 2.0],
    [7.0, 8.0],
    [13.0, 3.0],
    [17.0, 9.0]
])

# Initial bee position
bee = np.array([2.0, 5.0])

# Particle filter hypotheses
particles = rng.uniform(
    [0, 0], [W, H], (N_FILTER, 2)
)
weights = np.ones(N_FILTER) / N_FILTER

# Physical pollen grains
sources = flowers[rng.integers(len(flowers), size=N_POLLEN)]
pollen = sources + rng.normal(0, 0.4, (N_POLLEN, 2))

actual_path = []
estimated_path = []
errors = []

# Flower visitation
target_index = 0
visits = np.zeros(len(flowers), dtype=int)
carrying_pollen = False

# ============================================================
# WIND FIELD
# ============================================================

def wind(x, y, t):
    wx = 0.55 + 0.30 * np.sin(y * 0.6 + t * 0.08)
    wy = 0.28 * np.cos(x * 0.45 - t * 0.06)
    return np.broadcast_arrays(wx, wy)


# ============================================================
# PARTICLE FILTER RESAMPLING
# ============================================================

def systematic_resample(w):
    positions = (rng.random() + np.arange(len(w))) / len(w)
    cumulative = np.cumsum(w)
    cumulative[-1] = 1.0
    return np.searchsorted(cumulative, positions)


# ============================================================
# BEE DRAWING
# ============================================================

def draw_bee(ax, x, y, direction):
    """Draw a recognizable bee using Matplotlib patches."""

    angle = np.degrees(np.arctan2(direction[1], direction[0]))

    # Wings
    for offset in [-0.25, 0.25]:
        wing = Ellipse(
            (x - 0.1, y + offset),
            width=0.55,
            height=0.35,
            angle=angle,
            facecolor="#d8f5ff",
            edgecolor="#679ca8",
            alpha=0.9,
            zorder=19
        )
        ax.add_patch(wing)

    # Body
    body = Ellipse(
        (x, y),
        width=0.85,
        height=0.48,
        angle=angle,
        facecolor="#ffcf33",
        edgecolor="#30241c",
        linewidth=2,
        zorder=20
    )
    ax.add_patch(body)

    # Black stripes
    heading = np.array([
        np.cos(np.radians(angle)),
        np.sin(np.radians(angle))
    ])

    for shift in [-0.12, 0.12]:
        stripe_position = np.array([x, y]) + shift * heading
        stripe = Ellipse(
            stripe_position,
            width=0.12,
            height=0.44,
            angle=angle,
            facecolor="#30241c",
            zorder=21
        )
        ax.add_patch(stripe)

    # Head
    head_position = np.array([x, y]) + 0.42 * heading

    head = Circle(
        head_position,
        radius=0.20,
        facecolor="#30241c",
        edgecolor="white",
        linewidth=0.5,
        zorder=22
    )
    ax.add_patch(head)


# ============================================================
# FIGURE SETUP
# ============================================================

fig, ax = plt.subplots(figsize=(11, 7))
fig.patch.set_facecolor("#f8fbf4")

# ============================================================
# ANIMATION
# ============================================================

def animate(frame):
    global bee, particles, weights
    global pollen, target_index, carrying_pollen

    t = frame * DT

    # --------------------------------------------------------
    # 1. BEE MOVEMENT TOWARD FLOWERS
    # --------------------------------------------------------

    target = flowers[target_index]
    delta = target - bee
    distance = np.linalg.norm(delta)

    if distance < 0.45:
        visits[target_index] += 1
        carrying_pollen = True
        target_index = (target_index + 1) % len(flowers)

        target = flowers[target_index]
        delta = target - bee
        distance = np.linalg.norm(delta)

    direction = delta / max(distance, 1e-8)

    # Bee-directed velocity
    command = 2.2 * direction

    # Wind disturbance
    wx, wy = wind(bee[0], bee[1], frame)
    drift = 0.25 * np.array([float(wx), float(wy)])

    bee += (command + drift) * DT
    bee += rng.normal(0, 0.015, 2)
    bee = np.clip(bee, [0, 0], [W, H])

    # --------------------------------------------------------
    # 2. POLLEN PHYSICS
    # --------------------------------------------------------

    px, py = wind(pollen[:, 0], pollen[:, 1], frame)

    diffusion = 0.13
    settling = 0.08

    pollen[:, 0] += px * DT
    pollen[:, 1] += (py - settling) * DT

    pollen += rng.normal(
        0,
        np.sqrt(2 * diffusion * DT),
        pollen.shape
    )

    outside = (
        (pollen[:, 0] < 0) |
        (pollen[:, 0] > W) |
        (pollen[:, 1] < 0) |
        (pollen[:, 1] > H)
    )

    count = np.count_nonzero(outside)

    if count:
        new_sources = flowers[
            rng.integers(len(flowers), size=count)
        ]
        pollen[outside] = (
            new_sources + rng.normal(0, 0.2, (count, 2))
        )

    # --------------------------------------------------------
    # 3. PARTICLE FILTER PREDICTION
    # --------------------------------------------------------

    pwx, pwy = wind(
        particles[:, 0],
        particles[:, 1],
        frame
    )

    particle_drift = (
        np.column_stack([pwx, pwy]) * 0.25
    )

    particles += (command + particle_drift) * DT
    particles += rng.normal(0, 0.09, particles.shape)
    particles = np.clip(particles, [0, 0], [W, H])

    # --------------------------------------------------------
    # 4. NOISY SENSOR MEASUREMENTS
    # --------------------------------------------------------

    measurements = np.linalg.norm(
        flowers - bee, axis=1
    )

    measurements += rng.normal(
        0, 0.35, len(flowers)
    )

    predicted_ranges = np.linalg.norm(
        particles[:, None, :] - flowers[None, :, :],
        axis=2
    )

    # --------------------------------------------------------
    # 5. BAYESIAN WEIGHT UPDATE
    # --------------------------------------------------------

    residual = predicted_ranges - measurements

    log_w = -0.5 * np.sum(
        (residual / 0.35) ** 2,
        axis=1
    )

    log_w -= np.max(log_w)

    weights = np.exp(log_w)
    weights /= np.sum(weights)

    estimate = np.average(
        particles, axis=0, weights=weights
    )

    error = np.linalg.norm(estimate - bee)

    actual_path.append(bee.copy())
    estimated_path.append(estimate.copy())
    errors.append(error)

    # --------------------------------------------------------
    # 6. RESAMPLING
    # --------------------------------------------------------

    indices = systematic_resample(weights)

    particles = particles[indices].copy()
    weights.fill(1.0 / N_FILTER)

    # --------------------------------------------------------
    # 7. VISUALIZATION
    # --------------------------------------------------------

    ax.clear()
    ax.set_facecolor("#f8fbf4")
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.set_aspect("equal")
    ax.grid(alpha=0.12)

    # Wind arrows
    gx, gy = np.meshgrid(
        np.arange(1, W, 2.5),
        np.arange(1, H, 2.5)
    )

    ux, uy = wind(gx, gy, frame)

    ax.quiver(
        gx, gy, ux, uy,
        color="#83b7c4",
        alpha=0.45,
        scale=12,
        label="Wind"
    )

    # Flowers
    ax.scatter(
        flowers[:, 0],
        flowers[:, 1],
        c="#dd6496",
        s=350,
        marker="*",
        edgecolors="#9b3663",
        label="Flowers",
        zorder=5
    )

    # Pollen grains
    ax.scatter(
        pollen[:, 0],
        pollen[:, 1],
        c="#e4ae28",
        s=13,
        alpha=0.65,
        label="Pollen grains"
    )

    # Particle filter hypotheses
    ax.scatter(
        particles[:, 0],
        particles[:, 1],
        c="#7770c5",
        s=8,
        alpha=0.20,
        label="Filter particles"
    )

    # Bee trajectory
    actual = np.array(actual_path)
    estimated = np.array(estimated_path)

    ax.plot(
        actual[:, 0],
        actual[:, 1],
        color="#26906c",
        linewidth=2.5,
        alpha=0.75,
        label="True path"
    )

    # Estimated trajectory
    ax.plot(
        estimated[:, 0],
        estimated[:, 1],
        color="#c44f56",
        linewidth=1.8,
        linestyle="--",
        alpha=0.85,
        label="Estimated path"
    )

    # Estimated bee location
    ax.scatter(
        estimate[0],
        estimate[1],
        c="#d74e55",
        s=180,
        marker="x",
        linewidths=3,
        label="Estimated bee",
        zorder=25
    )

    # Draw visible bee
    draw_bee(ax, bee[0], bee[1], direction)

    # Highlight target flower
    ax.scatter(
        target[0],
        target[1],
        s=650,
        facecolors="none",
        edgecolors="#26906c",
        linewidths=2,
        linestyle="--",
        zorder=6
    )

    # Information panel
    ax.text(
        0.98, 0.04,
        f"Target flower: {target_index + 1}\n"
        f"Flower visits: {np.sum(visits)}\n"
        f"Localization error: {error:.2f} m",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=10,
        bbox=dict(
            boxstyle="round,pad=0.6",
            facecolor="white",
            edgecolor="#d7e6d9",
            alpha=0.9
        )
    )

    ax.set_title(
        "Bee Pollination and Particle Filter Localization\n"
        f"Simulation time: {t:.1f} s",
        fontsize=14,
        fontweight="bold",
        color="#245b49"
    )

    ax.legend(
        loc="upper left",
        fontsize=8,
        ncol=2
    )

    return []


# ============================================================
# EXPORT GIF
# ============================================================

animation = FuncAnimation(
    fig,
    animate,
    frames=FRAMES,
    interval=1000 / FPS,
    repeat=False,
    cache_frame_data=False
)

output = Path(__file__).resolve().parent / "bee_particle_filter.gif"

animation.save(
    str(output),
    writer=PillowWriter(fps=FPS),
    dpi=85
)

print(f"Saved GIF: {output}")
print(f"Mean localization error: {np.mean(errors):.3f} m")
print(f"Total flower visits: {np.sum(visits)}")

plt.close(fig)
