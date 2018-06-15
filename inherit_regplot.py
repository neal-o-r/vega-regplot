import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt


def Regplot_factory(base):
        class Regplot(base):
                def __init__(self, n_boot=100, ci=95, **kwargs):
                        super().__init__(self, **kwargs)
                        self.which = ci
                        self.n_boot = n_boot

                def _percentiles(self, a, pcts):
                        scores = []
                        for i, p in enumerate(pcts):
                                score = np.apply_along_axis(
                                                stats.scoreatpercentile, 0, a, p)
                                scores.append(score)

                        scores = np.asarray(scores)
                        return scores

                def _ci(self, a):
                        p = (50 - self.which / 2, 50 + self.which / 2)
                        return self._percentiles(a, p)

                def _bootstrap(self, x, y, xs):
                        n = len(x)
                        boot_dist = []
                        for i in range(int(self.n_boot)):
                                resampler = np.random.randint(0, n, n)
                                sample = [a.take(resampler, axis=0) for a in (x, y)]
                                self.fit(*sample)
                                boot_dist.append(self.predict(xs).reshape(-1,1))

                        return np.array(boot_dist)


                def plot(self, x, y, xs, ax=None,
                        scatter_args={'color':'k', 'alpha':0.6, 'marker':'.'},
                        line_args={'color':'#4286f4', 'lw':2.},
                        fill_args={'color':'#4286f4', 'alpha':0.3}):
                        '''
                        Do plotting. Give x and y to fit to, and xs where you want
                        predictions. ax object optional, if given it will plot in
                        that ax. 3 dicts of args accepted, one for scatter,
                        one for line, one for fill
                        '''

                        if ax is None:
                                fig, ax = plt.subplots()

                        self.fit(x, y)
                        yhat = self.predict(xs)

                        ci1, ci2 = self._ci(self._bootstrap(x, y, xs))[:, :, 0]
                        ax.scatter(x, y, **scatter_args)
                        ax.fill_between(xs.ravel(), ci1, ci2, **fill_args)
                        ax.plot(xs, yhat, **line_args,
                                        label=f'{self.score(x, y):.2f}')
                        ax.legend(loc='best')

                        return ax

        return Regplot

Regplot = Regplot_factory(LinearRegression)

