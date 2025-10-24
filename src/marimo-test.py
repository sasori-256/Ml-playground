import marimo

__generated_with = "0.16.5"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    from random import randint
    import numpy as np
    return np, randint


@app.cell
def _(randint):
    idx_students = [randint(0, 50) for i in range(5)]
    return (idx_students,)


@app.cell
def _(idx_students):
    print(idx_students)
    return


@app.cell
def _(np):
    students = np.array([170, 172, 167, 173, 172])
    print(students.mean())
    print(np.median(students, axis=0))
    sum_of_squares = sum(students**2) - (sum(students)**2) / len(students)

    print(sum_of_squares / len(students))
    print(np.var(students, ddof=0))
    print(sum_of_squares / (len(students) - 1))
    print(np.var(students, ddof=1))
    return


@app.cell
def _(randint):
    idx_get_up_time = [randint(1, 30) for i in range(6)]
    return (idx_get_up_time,)


@app.cell
def _(idx_get_up_time):
    idx_get_up_time
    return


@app.cell
def _(np):
    get_up_time = [8, 8, 7, 6, 7, 8]
    is_before_7 = [1 if x < 7 else 0 for x in get_up_time]
    print(np.mean(is_before_7))
    sum_g = np.sum(is_before_7)
    len_g = len(is_before_7)
    print((1 + sum_g) / (len_g + 2))
    print((1/2 + sum_g) / (len_g + 1))
    return


@app.cell
def _(randint):
    idx_downtime = [randint(1, 40) for i in range(3)]
    return (idx_downtime,)


@app.cell
def _(idx_downtime):
    idx_downtime
    return


@app.cell
def _(np):
    downtime = [1454, 1491, 2752]
    print(np.mean(downtime))
    return


@app.cell
def _(randint):
    idx_visitors_num = [randint(1, 70) for _ in range(10)]
    return (idx_visitors_num,)


@app.cell
def _(idx_visitors_num):
    idx_visitors_num
    return


@app.cell
def _(np):
    visitors_num = [15, 12, 12, 11, 9, 9, 14, 14, 12, 10]
    print(np.mean(visitors_num))
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
