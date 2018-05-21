import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from bokeh.plotting import figure, show, output_file
from bokeh.models import HoverTool
from seaborn.regression import _RegressionPlotter

tips = sns.load_dataset("tips")
tips.sort_values(by='total_bill', inplace=True)

regplot = _RegressionPlotter('total_bill', 'tip', data=tips)
grid, yhat, err_bands = regplot.fit_regression(grid=tips.total_bill)

tips['yhat'] = yhat
tips['ci1'] = err_bands[0]
tips['ci2'] = err_bands[1]

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

