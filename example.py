from regplot import Regplot
import seaborn as sns # only using this for the dataset
import matplotlib.pyplot as plt
from sklearn.linear_model import HuberRegressor
import numpy as np

if __name__ == '__main__':

        df = sns.load_dataset("tips")
        x = df.total_bill.values.reshape(-1,1)
        y = df.tip.values.reshape(-1,1)
        xs = np.linspace(x.min(), x.max(), 100).reshape(-1,1)

        regplotter = Regplot()
        regplotter.plot(x, y, xs)
        plt.show()

        # huber regressor require a FLATTENED Y (this is really stupid)
        # and throws a really weird warning that I don't understand
        regplotter = Regplot(regressor=HuberRegressor)
        regplotter.plot(x, y.ravel(), xs)
        plt.show()
