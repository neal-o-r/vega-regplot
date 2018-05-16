import seaborn as sns; sns.set(color_codes=True)
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from bokeh.plotting import figure, show, output_file
from bokeh.models import HoverTool

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


def basis(x, deg):
        return np.vander(x, deg+1)


def solve(x, y, deg=1):
        x = basis(x, deg)
        return np.linalg.pinv(x).dot(y.reshape(-1,1))


def predict(x, w):
        return basis(x, 1).dot(w)

def fit(x, y):
        w = solve(x, y)
        return predict(x, w)

tips.sort_values(by='total_bill', inplace=True)
tips['yhat'] = fit(tips.total_bill.values, tips.tip.values)
ci1, ci2 = ci(bootstrap(tips.total_bill.values, tips.tip.values))[:, :, 0]

tips['ci1'] = ci1
tips['ci2'] = ci2

hover = HoverTool(tooltips=[
    ("(x, y)", "($x, $y)"),
])
tools = [hover, 'pan', 'wheel_zoom']

p = figure(title="Bokeh Regplot", toolbar_location='right', tools=tools)

p.scatter('total_bill', 'tip', source=tips)
p.line('total_bill', 'yhat', source=tips, line_width=2, line_color='grey')
p.line('total_bill', 'ci1', source=tips, alpha=0.7,
                            line_color='grey', line_dash='dashed')
p.line('total_bill', 'ci2', source=tips, alpha=0.7,
                            line_color='grey', line_dash='dashed')

p.xaxis.axis_label = "total_bill"
p.yaxis.axis_label = "tip"

show(p)

