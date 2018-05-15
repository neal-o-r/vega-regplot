import seaborn as sns; sns.set(color_codes=True)
import matplotlib.pyplot as plt
import altair as alt
import numpy as np
from scipy import stats

tips = sns.load_dataset("tips")
def percentiles(a, pcts):
        scores = []
        for i, p in enumerate(pcts):
                score = np.apply_along_axis(stats.scoreatpercentile, 0, a, p)
                scores.append(score)
        scores = np.asarray(scores)
        return scores


def ci(a, which=95):
        p = (50 - which / 2, 50 + which / 2)
        return percentiles(a, p)


def bootstrap(x, y, n_boot=100):
        n = len(x)
        boot_dist = []
        for i in range(int(n_boot)):
                resampler = np.random.randint(0, n, n)
                sample = [a.take(resampler, axis=0) for a in (x, y)]
                w = solve(*sample)
                boot_dist.append(predict(x, w))

        return np.array(boot_dist)


def basis(x, deg=1):
        return np.vander(x, deg+1)


def solve(x, y):
        x = basis(x)
        return np.linalg.pinv(x).dot(y.reshape(-1,1))


def predict(x, w):
        return basis(x).dot(w)

def fit(x, y):
        w = solve(x, y)
        return predict(x, w)


sns.regplot(x="total_bill", y="tip", data=tips)
plt.show()

tips['yhat'] = fit(tips.total_bill.values, tips.tip.values)
ci1, ci2 = ci(bootstrap(tips.total_bill.values, tips.tip.values))[:, :, 0]

tips['ci1'] = ci1
tips['ci2'] = ci2


points = alt.Chart(tips).mark_point().encode(
    x='total_bill',
    y='tip'
).interactive()

line = alt.Chart(tips).mark_line(color='black').encode(
        x='total_bill',
        y=alt.Y('yhat', axis=alt.Axis(title='tip'))
)

area = alt.Chart(tips).mark_area(opacity=0.3, color='grey').encode(
        x='total_bill',
        y='ci1',
        y2='ci2'
)

chart = points + line + area

chart.save('regplot.html')
