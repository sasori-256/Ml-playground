import marimo

__generated_with = "0.16.5"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import scipy.stats as st
    from typing import List
    from math import sqrt
    return List, np, sqrt


@app.cell
def _(np):
    prod_a = np.array([36, 264, 150])
    prod_b = np.array([113, 187, 120, 95, 205, 180])
    print(np.std(prod_a, ddof=1))
    print(np.std(prod_b, ddof=1))
    return prod_a, prod_b


@app.cell
def _(List, sqrt):
    # 2-(a)
    num_a = 25
    std_a = 5
    mean_a = 30

    # 平均に関する信頼区間
    # 95%
    conf_a95: float = 1.960
    interval_a95: List[float] = [
        mean_a - conf_a95 * std_a / sqrt(num_a),
        mean_a + conf_a95 * std_a / sqrt(num_a),
    ]
    print(interval_a95)

    # 99%
    conf_a99: float = 2.576
    interval_a99: List[float] = [
        mean_a - conf_a99 * std_a / sqrt(num_a),
        mean_a + conf_a99 * std_a / sqrt(num_a),
    ]
    print(interval_a99)
    return


@app.cell
def _(List, sqrt):
    # 2-(b)
    num_b = 100
    std_b = 5
    mean_b = 30

    # 平均に関する信頼区間
    # 95%
    conf_b95: float = 1.960
    interval_b95: List[float] = [
        mean_b - conf_b95 * std_b / sqrt(num_b),
        mean_b + conf_b95 * std_b / sqrt(num_b),
    ]
    print(interval_b95)

    # 99%
    conf_b99: float = 2.576
    interval_b99: List[float] = [
        mean_b - conf_b99 * std_b / sqrt(num_b),
        mean_b + conf_b99 * std_b / sqrt(num_b),
    ]
    print(interval_b99)
    return


@app.cell
def _(List, sqrt):
    # 2-(c)
    num_c = 1000
    std_c = 5
    mean_c = 30

    # 平均に関する信頼区間
    # 95%
    conf_c95: float = 1.960
    interval_c95: List[float] = [
        mean_c - conf_c95 * std_c / sqrt(num_c),
        mean_c + conf_c95 * std_c / sqrt(num_c),
    ]
    print(interval_c95)

    # 99%
    conf_c99: float = 2.576
    interval_c99: List[float] = [
        mean_c - conf_c99 * std_c / sqrt(num_c),
        mean_c + conf_c99 * std_c / sqrt(num_c),
    ]
    print(interval_c99)
    return


@app.cell
def _(List, sqrt):
    # 3
    num_3 = 15
    std_3 = sqrt(183.75)
    mean_3 = 30

    # 平均に関する信頼区間
    # 95%
    conf_395: float = 2.145
    interval_395: List[float] = [
        mean_3 - conf_395 * std_3 / sqrt(num_3),
        mean_3 + conf_395 * std_3 / sqrt(num_3),
    ]
    print(interval_395)

    # 99%
    conf_399: float = 2.977
    interval_399: List[float] = [
        mean_3 - conf_399 * std_3 / sqrt(num_3),
        mean_3 + conf_399 * std_3 / sqrt(num_3),
    ]
    print(interval_399)
    return


@app.cell
def _(List, sqrt):
    # 4
    num_4 = 100
    mean_4 = 0.6
    std_4 = sqrt(mean_4 * (1 - mean_4))

    # 平均に関する信頼区間
    # 95%
    conf_495: float = 1.960
    interval_495: List[float] = [
        mean_4 - conf_495 * std_4 / sqrt(num_4),
        mean_4 + conf_495 * std_4 / sqrt(num_4),
    ]
    print(interval_495)

    # 99%
    conf_499: float = 2.576
    interval_499: List[float] = [
        mean_4 - conf_499 * std_4 / sqrt(num_4),
        mean_4 + conf_499 * std_4 / sqrt(num_4),
    ]
    print(interval_499)
    return


@app.cell
def _(List, np, prod_a, sqrt):
    # 5-a
    num_5a = 3
    mean_5a = np.mean(prod_a)
    std_5a = np.std(prod_a, ddof=1)

    # 平均に関する信頼区間
    # 95%
    conf_5a95: float = 1.960
    interval_5a95: List[float] = [
        mean_5a - conf_5a95 * std_5a / sqrt(num_5a),
        mean_5a + conf_5a95 * std_5a / sqrt(num_5a),
    ]
    print(interval_5a95)

    # 99%
    conf_5a99: float = 2.576
    interval_5a99: List[float] = [
        mean_5a - conf_5a99 * std_5a / sqrt(num_5a),
        mean_5a + conf_5a99 * std_5a / sqrt(num_5a),
    ]
    print(interval_5a99)
    return


@app.cell
def _(List, np, prod_b, sqrt):
    # 5-b
    num_5b = 6
    mean_5b = np.mean(prod_b)
    std_5b = np.std(prod_b, ddof=1)

    # 平均に関する信頼区間
    # 95%
    conf_5b95: float = 1.960
    interval_5b95: List[float] = [
        mean_5b - conf_5b95 * std_5b / sqrt(num_5b),
        mean_5b + conf_5b95 * std_5b / sqrt(num_5b),
    ]
    print(interval_5b95)

    # 99%
    conf_5b99: float = 2.576
    interval_5b99: List[float] = [
        mean_5b - conf_5b99 * std_5b / sqrt(num_5b),
        mean_5b + conf_5b99 * std_5b / sqrt(num_5b),
    ]
    print(interval_5b99)
    return


@app.cell
def _(List, np, sqrt):
    # 6
    traffic = np.array([8, 5, 10, 5, 15, 9, 6, 18, 8, 13])
    num_6 = len(traffic)
    mean_6 = np.mean(traffic)
    std_6 = sqrt(mean_6)

    # 平均に関する信頼区間
    # 95%
    conf_695: float = 1.960
    interval_695: List[float] = [
        mean_6 - conf_695 * std_6 / sqrt(num_6),
        mean_6 + conf_695 * std_6 / sqrt(num_6),
    ]
    print(interval_695)

    # 99%
    conf_699: float = 2.576
    interval_699: List[float] = [
        mean_6 - conf_699 * std_6 / sqrt(num_6),
        mean_6 + conf_699 * std_6 / sqrt(num_6),
    ]
    print(interval_699)
    return


@app.cell
def _(List):
    # 7
    num_7 = 15
    mean_7 = 30
    var_7 = 183.75

    # 平均に関する信頼区間
    # 95%
    conf_7_95_above: float = 26.12
    conf_7_95_below: float = 5.629

    interval_7_95: List[float] = [var_7 / conf_7_95_above, var_7 / conf_7_95_below]
    print(interval_7_95)

    # 99%
    conf_7_99_above: float = 31.32
    conf_7_99_below: float = 4.075
    interval_7_99: List[float] = [var_7 / conf_7_99_above, var_7 / conf_7_99_below]
    print(interval_7_99)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
