"""Example how to generate random approval profiles."""

from abcvoting import generate
from abcvoting.generate import PointProbabilityDistribution
import matplotlib.pyplot as plt


# distributions to generate points in 1- and 2-dimensional space
distributions = [
    PointProbabilityDistribution("1d_interval", center_point=[0]),
    PointProbabilityDistribution("1d_gaussian", center_point=[4]),
    PointProbabilityDistribution("1d_gaussian_interval", center_point=[6], width=0.5),
    PointProbabilityDistribution("2d_square", center_point=[0, 2]),
    PointProbabilityDistribution("2d_disc", center_point=[2, 2]),
    PointProbabilityDistribution("2d_gaussian", center_point=[4, 2], sigma=0.25),
    PointProbabilityDistribution("2d_gaussian_disc", center_point=[6, 2], sigma=0.25),
]

fig, ax = plt.subplots(dpi=600, figsize=(8, 3))
points = []
for dist in distributions:
    if dist.name.startswith("2d"):
        for _ in range(1000):
            points.append(generate.random_point(dist))
        title_coord = [dist.center_point[0], dist.center_point[1] + 0.6]
    else:
        for _ in range(100):
            points.append([generate.random_point(dist), 0])
        title_coord = [dist.center_point[0], 0.2]
    title = dist.name + "\n"
    if dist.width != 1.0:
        title += f"(width={dist.width})"
    plt.annotate(title, title_coord, ha="center")

ax.scatter([x for x, y in points], [y for x, y in points], alpha=0.5, s=5)
plt.gca().set_aspect("equal", adjustable="box")
plt.xlim(-0.8, 7.3)
plt.ylim(-0.2, 3.2)
fig.tight_layout()
plt.show()
