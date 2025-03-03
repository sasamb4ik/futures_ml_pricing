import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm_notebook as tqdm
from sklearn.base import TransformerMixin
from IPython.display import display, clear_output
from dataclasses import dataclass

from abstracts import SamplerAbstract, PricerAbstract

def _plot_progress(sampler, bar, price_history, lower_bound, upper_bound):
    clear_output(wait=True)
    display(bar.container)
    plt.ticklabel_format(style='plain', useOffset=False)
    plt.plot(sampler.time_grid, price_history, label='price')
    plt.plot(sampler.time_grid, lower_bound, "--", label='lower bound')
    plt.plot(sampler.time_grid, upper_bound, "--", label="upper bound")
    plt.legend()
    plt.title("$Option_t$")
    plt.xlabel("$t$")
    plt.ylabel("price")
    plt.grid()
    plt.show()

@dataclass
class AmericanMonteCarloResult:
    price_history: np.ndarray
    lower_bound: np.ndarray
    upper_bound: np.ndarray

class PricerAmericanMonteCarlo(PricerAbstract):
    def __init__(
            self,
            sampler: SamplerAbstract,
            basis_functions_transformer: TransformerMixin,
            regularization_alpha: float = 1e-4
    ):
        self.sampler = sampler
        self.regularization_alpha = regularization_alpha
        self.basis_functions_transformer = basis_functions_transformer
        self.price_history: np.ndarray | None = None
        self.option_price: np.ndarray | None = None
        self.result = {}

    def price(self, test=False, quiet=False):
        self.sampler.sample()
        if not quiet:
            self.sampler.plot(cnt=10, plot_mean=True, y="payoff, discount_factor, markov_state")
    
    # Вычисление дисконтированного payoff
        discounted_payoff = self.sampler.payoff * self.sampler.discount_factor
    
    # Инициализация option_price как payoff в последний момент времени
        self.option_price = discounted_payoff[:, -1].copy()
        weights = [None] * self.sampler.cnt_times
        self.price_history = [None] * (self.sampler.cnt_times - 1) + [self.option_price.mean()]

    # Вычисление upper_bound
        upper_bound = np.zeros(self.sampler.cnt_times)
        for i in range(self.sampler.cnt_times):
            upper_bound[i] = discounted_payoff[:, i:].max(axis=1).mean()

        bar = tqdm(range(self.sampler.cnt_times - 2, -1, -1))
        for time_index in bar:
        # Вычисление продолжительной стоимости на всех траекториях
            features = self.sampler.markov_state[:, time_index]
            transformed = self.basis_functions_transformer.fit_transform(features)
            regularization = np.eye(transformed.shape[1], dtype=float) * self.regularization_alpha
            inv = np.linalg.pinv((transformed.T @ transformed + regularization), rcond=1e-4)
            weights[time_index] = inv @ transformed.T @ self.option_price
            continuation_value = transformed @ weights[time_index]

        # Обновление option_price для всех траекторий
            indicator = discounted_payoff[:, time_index] > continuation_value
            self.option_price = (indicator * discounted_payoff[:, time_index] +
                            ~indicator * self.option_price)
        
        # Обновление price_history
            self.price_history[time_index] = self.option_price.mean()
        
            if not quiet and time_index % 10 == 0:
                _plot_progress(self.sampler, bar, self.price_history, self.price_history, upper_bound)

        if not quiet:
            self.sampler.plot(cnt=10, plot_mean=True, y="payoff, discount_factor, markov_state")

        key = "test" if test else "train"
        self.result[key] = {
            "price": float(self.option_price.mean()),
            "upper_bound": float(discounted_payoff.max(axis=1).mean()),
            "lower_bound": float(self.price_history[0]),
            "std": float(self.option_price.std())
        }

        return self.price_history