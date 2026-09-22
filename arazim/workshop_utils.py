"""Plotting helpers for the Vistula ML workshop.

Nothing in this file computes a gradient or trains anything -- that all stays
visible in the notebooks. This is only here so the notebooks are not 60%
matplotlib boilerplate. Feel free to open it, there is nothing clever inside.
"""

import numpy as np
import matplotlib.pyplot as plt


def plot_function(f, xlim=(-3, 3), ax=None, n=200, **kw):
    """Draw a 1-D function f over xlim."""
    ax = ax or plt.gca()
    xs = np.linspace(xlim[0], xlim[1], n)
    ax.plot(xs, [f(x) for x in xs], color="0.6", lw=2, **kw)
    ax.set_xlabel("x")
    ax.set_ylabel("f(x)")
    return ax


def plot_path_1d(f, path, ax=None, color="crimson"):
    """Draw the descent path as dots sitting on the curve."""
    ax = ax or plt.gca()
    path = np.asarray(path)
    ax.plot(path, [f(x) for x in path], "o-", color=color, ms=5, lw=1, alpha=0.8)
    ax.plot(path[0], f(path[0]), "o", color="black", ms=9, label="start")
    ax.plot(path[-1], f(path[-1]), "*", color="gold", ms=18,
            mec="black", label="end")
    ax.legend(loc="upper center", fontsize=8)
    return ax


def plot_contour(f, xlim=(-3, 3), ylim=(-3, 3), ax=None, n=120, levels=20):
    """Contour map of a function taking a length-2 array."""
    ax = ax or plt.gca()
    gx, gy = np.meshgrid(np.linspace(*xlim, n), np.linspace(*ylim, n))
    zz = np.array([f(np.array([a, b]))
                   for a, b in zip(gx.ravel(), gy.ravel())]).reshape(gx.shape)
    ax.contourf(gx, gy, zz, levels=levels, cmap="Blues_r", alpha=0.9)
    ax.contour(gx, gy, zz, levels=levels, colors="white", linewidths=0.4)
    # Equal scaling on both axes: otherwise the contours get squashed and the
    # gradient no longer *looks* perpendicular to them, even though it is.
    ax.set_aspect("equal", adjustable="box")
    return ax


def plot_path_2d(path, ax=None, color="crimson"):
    """Draw a 2-D descent path on top of a contour map."""
    ax = ax or plt.gca()
    path = np.asarray(path)
    ax.plot(path[:, 0], path[:, 1], "o-", color=color, ms=4, lw=1.2, alpha=0.9)
    ax.plot(path[0, 0], path[0, 1], "o", color="black", ms=9)
    ax.plot(path[-1, 0], path[-1, 1], "*", color="gold", ms=18, mec="black")
    return ax


def plot_points(X, y, ax=None):
    """Scatter a 2-D dataset, coloured by the 0/1 label."""
    ax = ax or plt.gca()
    y = np.asarray(y).ravel()
    ax.scatter(X[y == 0, 0], X[y == 0, 1], c="#1f77b4", edgecolor="white",
               s=45, label="class 0", zorder=3)
    ax.scatter(X[y == 1, 0], X[y == 1, 1], c="#d62728", edgecolor="white",
               s=45, marker="s", label="class 1", zorder=3)
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    return ax


def plot_boundary(predict, X, y, ax=None, n=200, title=None):
    """Colour the plane by `predict` (a function of an (N, 2) array), then
    scatter the data on top. This is how we *see* what a neuron has learned."""
    ax = ax or plt.gca()
    pad = 0.6
    gx, gy = np.meshgrid(
        np.linspace(X[:, 0].min() - pad, X[:, 0].max() + pad, n),
        np.linspace(X[:, 1].min() - pad, X[:, 1].max() + pad, n))
    grid = np.c_[gx.ravel(), gy.ravel()]
    zz = np.asarray(predict(grid)).reshape(gx.shape)
    ax.contourf(gx, gy, zz, levels=np.linspace(0, 1, 21),
                cmap="coolwarm", alpha=0.75, vmin=0, vmax=1)
    ax.contour(gx, gy, zz, levels=[0.5], colors="black", linewidths=2)
    plot_points(X, y, ax=ax)
    if title:
        ax.set_title(title, fontsize=10)
    return ax
