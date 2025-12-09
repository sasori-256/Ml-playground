import marimo

__generated_with = "0.17.8"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import math
    return math, np


@app.cell
def _(np):
    data = np.array([
        52.1, 52.0, 52.2, 52.0, 52.1, 52.0, 52.1, 52.2, 52.3, 52.2,
        52.0, 52.2, 52.1, 52.0, 51.9, 52.1, 52.4, 52.2, 52.1, 52.2
    ])
    mean = np.mean(data)
    var = np.var(data, ddof=1)
    std = np.std(data, ddof=1)
    return data, mean, std, var


@app.cell
def _():
    print(0.18 ** 2)
    return


@app.cell
def _(mean, std, var):
    print(f"Mean: {mean}\nVar: {var}\nStd: {std}")
    return


@app.cell
def _(data, mean, np):
    deviation = data - mean
    squared_deviation = deviation ** 2
    sum_of_squares = np.sum(squared_deviation)

    standardization = sum_of_squares / (0.18 ** 2)
    print(f"Standardization: {standardization}")
    return (sum_of_squares,)


@app.cell
def _(sum_of_squares):
    sum_of_squares / 8.91
    return


@app.cell
def _(sum_of_squares):
    sum_of_squares / 32.9
    return


@app.cell
def _(math, np):
    poison_data = np.array([8, 5, 10, 5, 15, 9, 6, 18, 8, 13])
    poison_mean = np.mean(poison_data)

    poison_standardization = (poison_mean - 10) * math.sqrt(len(poison_data)) / math.sqrt(10)
    print(f"Standardization: {poison_standardization}")
    return (poison_mean,)


@app.cell
def _(poison_mean):
    print(poison_mean)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
