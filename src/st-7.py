import marimo

__generated_with = "0.17.7"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    return (np,)


@app.cell
def _(np):
    num = np.array([12.3, 11.9, 12.2, 12.0, 12.4, 12.1, 12.3, 12.4, 12.2])
    mean: float = np.mean(num)
    std1: float = 0.21
    return mean, num, std1


@app.cell
def _(mean: float, np, num):
    print(mean)
    print(np.size(num))
    return


@app.cell
def _(mean: float, np, num, std1: float):
    norm1: float = (mean - 12.0) * np.sqrt(np.size(num)) / std1
    print(norm1)
    return


@app.cell
def _(mean: float, np, num):
    var = np.var(num, ddof=1)
    print(var)
    std2 = np.std(num, ddof=1)
    print(std2)
    norm2: float = (mean - 12.0) * np.sqrt(np.size(num)) / std2
    print(norm2)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
