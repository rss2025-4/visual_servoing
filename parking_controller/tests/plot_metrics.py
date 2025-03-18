import pickle
from pathlib import Path

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from test_park import metric_dict


def main():
    with open(Path(__file__).parent / "record_metric.pkl", "rb") as file:
        metrics: metric_dict = pickle.load(file)

    xs = []
    ys = []
    zs = []

    for (x, y), m in metrics.items():
        print(x, y)
        print(m)
        print()
        xs.append(x)
        ys.append(y)
        zs.append(m.time_took)

    fig = plt.figure()

    # Add a 3D subplot
    ax = fig.add_subplot(111, projection="3d")
    assert isinstance(ax, Axes3D)

    # Scatter plot
    ax.scatter(xs, ys, zs)  # type: ignore

    # Set labels
    ax.set_xlabel("cone_x")
    ax.set_ylabel("Y Label")
    ax.set_zlabel("Z Label")

    # Show the plot
    plt.show()
