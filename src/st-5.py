import marimo

__generated_with = "0.16.5"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import pandas as pd
    return


@app.cell
def _():
    heightf_n = 17
    heightf_mean = 157.0
    heightf_std = 4.6

    print(heightf_std**2 / 1**2 * (1.960)**2)
    return


@app.cell
def _():
    heightm_n = 15
    heightm_mean = 169.1
    heightm_std = 6.8

    print(heightm_std**2 / 1**2 * (1.960)**2)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
