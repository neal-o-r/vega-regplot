import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

class Regplot():
        '''
        Make a regression with CI based on sklearn regressors.
        At initialisation give
                :regressor, a regressor from sklearn with .fit
                        .predict and .score methods
                :kwargs, a dict of keyword args for regressor
                :n_boot, number of bootstraps used in computing CI, def 100
                :ci, CI% to compute, defaults to 95%

        Calling .plot creates plot and returns ax object
        '''

        def __init__(self, regressor=LinearRegression, kwargs={},
                                n_boot=100, ci=95):
                self.kwargs = kwargs
                self.regressor = regressor(**self.kwargs)
                self.regressor_type = regressor
                self.n_boot = n_boot
                self.which = ci

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
                        reg = self.regressor_type(**self.kwargs)
                        resampler = np.random.randint(0, n, n)
                        sample = [a.take(resampler, axis=0) for a in (x, y)]
                        reg.fit(*sample)
                        boot_dist.append(reg.predict(xs).reshape(-1,1))

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

                self.regressor.fit(x, y)
                yhat = self.regressor.predict(xs)

                ci1, ci2 = self._ci(self._bootstrap(x, y, xs))[:, :, 0]
                ax.scatter(x, y, **scatter_args)
                ax.fill_between(xs.ravel(), ci1, ci2, **fill_args)
                ax.plot(xs, yhat, **line_args,
                                label=f'{self.regressor.score(x, y):.2f}')
                ax.legend(loc='best')

                return ax

