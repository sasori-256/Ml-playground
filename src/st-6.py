import marimo

__generated_with = "0.17.6"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # mo
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    仮説検定はマニュアル通りにやればいいという訳ではない

    帰無仮説の棄却のみを成果とすべきではない

    第2の過誤の確率も評価すべき
    """)
    return


if __name__ == "__main__":
    app.run()
