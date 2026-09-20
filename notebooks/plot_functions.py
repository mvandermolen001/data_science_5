import matplotlib.pyplot as plt
import numpy as np
import warnings
from math import ceil

"""
This code was originally 
created by Dave Langers for the Advanced Datamining module of the Bioinformatics 
course at the Hanze.

It has since been adjusted for use in the Data Science 5 course.
"""

def scatter(xs, ys, *, model=None):
    """Plot 2D data and optionally show model predictions.

    Parameters
    ----------
    xs : array-like
        Feature values. Must contain two features.
    ys : array-like
        True outcomes.
    model : object, optional
        Classification or regression model with a ``predict`` method,
        and optionally ``predict_proba`` or ``decision_function``.
    """
    xs = np.asarray(xs)
    ys = np.asarray(ys)
    # Make ys two-dimensional
    if ys.ndim == 1:
        ys = ys[:, np.newaxis]
    # Extract the two features
    x1s = xs[:, 0]
    x2s = xs[:, 1]
    # Determine plotting range
    xlimit = ceil(1.05 * max(abs(x1s.min()),
            abs(x1s.max()),abs(x2s.min()),
            abs(x2s.max())))
    xgrid = np.linspace(-xlimit, xlimit, 129)
    # Generate the hue of model predictions over the plotting grid
    back = None
    if model is not None:
        grid = np.array([[x1, x2] for x2 in xgrid for x1 in xgrid])
        if hasattr(model, "predict_proba"):
            back = model.predict_proba(grid)
        elif hasattr(model, "predict"):
            back = model.predict(grid)
        if back is not None:
            back = np.asarray(back).reshape(len(xgrid), len(xgrid))

    # Create the plot
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    ylimit = ceil(max(abs(ys.min()), abs(ys.max())))
    data = ax.scatter(x1s,x2s,
        c=ys,edgecolors="w",
        cmap=plt.cm.RdYlBu,vmin=-ylimit,
        vmax=ylimit)

    if back is None:
        ax.set_facecolor("#F8F8F8")
    else:
        ax.imshow(back,origin="lower",
        extent=(-xlimit, xlimit, -xlimit, xlimit),
        vmin=-ylimit,vmax=ylimit,
        interpolation="bilinear",cmap=plt.cm.RdYlBu)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ax.contour(xgrid,xgrid,
                back,levels=[0],
                colors="k",linestyles="--",
                linewidths=1)

    # Format axes
        ax.set(aspect="equal",
            xlim=(-xlimit, xlimit),
            ylim=(-xlimit, xlimit),
            xlabel=r"$x_1$",ylabel=r"$x_2$")

        ax.grid(True, color="k", linestyle=":", linewidth=0.5)
        ax.axhline(0, color="k", linewidth=1)
        ax.axvline(0, color="k", linewidth=1)
        ax.set_axisbelow(True)

        # Colour bar
        cbar = fig.colorbar(data, ax=ax)
        cbar.ax.axhline(0, color="k", linestyle="--", linewidth=1)
        cbar.ax.set_title(r"$y$")

    plt.show()