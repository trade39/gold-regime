import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression

class GoldRegimeModel:
    def __init__(self, k_regimes=2):
        self.k_regimes = k_regimes
        self.model = None
        self.results = None
        
    def fit(self, returns):
        """
        Fits a Markov Switching Model to the returns.
        Assumes returns are already scaled (e.g., percentages).
        
        Args:
            returns (pd.Series): Time series of returns.
        """
        # Hamilton suggestion: switching mean and switching variance
        # "Focusing only on the most important shifting parameters, such as the intercept (mean return) and variance"
        self.model = MarkovRegression(returns, k_regimes=self.k_regimes, trend='c', switching_variance=True)
        
        try:
            self.results = self.model.fit()
        except Exception as e:
            print(f"Model fitting failed: {e}")
            raise e
            
        return self.results.summary()
        
    def predict_probs(self):
        """
        Returns the smoothed probabilities of being in each regime.
        """
        if self.results is None:
            raise ValueError("Model not fitted yet.")
        # smoothed_marginal_probabilities gives P(S_t=j | T) (using full sample)
        return self.results.smoothed_marginal_probabilities

    def get_regime_stats(self):
        """
        Returns parameters to help interpret regimes (e.g., High Vol vs Low Vol).
        """
        if self.results is None:
            raise ValueError("Model not fitted yet.")
            
        params = self.results.params
        stats = {}
        
        # Extract mean and variance for each regime
        # Statsmodels naming convention: 'const[0]', 'sigma2[0]', 'const[1]', 'sigma2[1]'
        for i in range(self.k_regimes):
            regime_name = f'Regime {i}'
            stats[regime_name] = {
                'Mean Return': params.get(f'const[{i}]', 0),
                'Variance': params.get(f'sigma2[{i}]', 0),
                'Volatility (Std Dev)': params.get(f'sigma2[{i}]', 0) ** 0.5
            }
        return pd.DataFrame(stats)
