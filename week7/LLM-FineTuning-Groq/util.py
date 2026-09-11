"""
Utility functions for model evaluation and visualization
"""

import re
import math
from typing import List, Callable, Tuple, Dict, Any
from tqdm.auto import tqdm
from IPython.display import clear_output
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import mean_squared_error, r2_score
from itertools import accumulate


# Color codes for terminal output
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
RESET = "\033[0m"
COLOR_MAP = {"red": RED, "orange": YELLOW, "green": GREEN}

DEFAULT_SIZE = 200


class Tester:
    """
    Evaluates model predictions against ground truth values
    Generates visualization charts and error metrics
    """
    
    def __init__(self, predictor: Callable, data: List[Dict[str, Any]], 
                 title: str = None, size: int = DEFAULT_SIZE):
        """
        Initialize Tester
        
        Args:
            predictor: Function that takes a datapoint and returns predicted value
            data: List of datapoints with prompt and completion fields
            title: Optional title for the tester
            size: Number of datapoints to evaluate
        """
        self.predictor = predictor
        self.data = data
        self.title = title or self.make_title(predictor)
        self.size = size
        self.titles = []
        self.guesses = []
        self.truths = []
        self.errors = []
        self.colors = []

    @staticmethod
    def make_title(predictor: Callable) -> str:
        """Generate title from predictor function name"""
        return (predictor.__name__
                .replace("__", ".")
                .replace("_", " ")
                .title()
                .replace("Gpt", "GPT")
                .replace("Groq", "Groq"))

    @staticmethod
    def post_process(value: Any) -> float:
        """
        Convert model output to numeric value
        Handles currency formatting and strings
        """
        if isinstance(value, str):
            value = value.replace("$", "").replace(",", "")
            match = re.search(r"[-+]?\d*\.\d+|\d+", value)
            return float(match.group()) if match else 0
        else:
            return float(value)

    def color_for(self, error: float, truth: float) -> str:
        """Determine color based on error magnitude"""
        if error < 40 or error / truth < 0.2:
            return "green"
        elif error < 80 or error / truth < 0.4:
            return "orange"
        else:
            return "red"

    def run_datapoint(self, i: int) -> Tuple[str, float, float, float, str]:
        """
        Evaluate single datapoint
        
        Returns:
            Tuple of (title, guess, truth, error, color)
        """
        datapoint = self.data[i]
        value = self.predictor(datapoint)
        guess = self.post_process(value)
        truth = float(datapoint["completion"])
        error = abs(guess - truth)
        color = self.color_for(error, truth)
        
        pieces = datapoint["prompt"].split("Title: ")
        title = pieces[1].split("\n")[0] if len(pieces) > 1 else pieces[0]
        title = title if len(title) <= 40 else title[:40] + "..."
        
        return title, guess, truth, error, color

    def chart(self, title: str) -> None:
        """Generate scatter plot of predictions vs actuals"""
        df = pd.DataFrame({
            "truth": self.truths,
            "guess": self.guesses,
            "title": self.titles,
            "error": self.errors,
            "color": self.colors,
        })

        # Pre-format hover text
        df["hover"] = [
            f"{t}\nGuess=${g:,.2f} Actual=${y:,.2f}"
            for t, g, y in zip(df["title"], df["guess"], df["truth"])
        ]

        max_val = float(max(df["truth"].max(), df["guess"].max()))

        fig = px.scatter(
            df,
            x="truth",
            y="guess",
            color="color",
            color_discrete_map={"green": "green", "orange": "orange", "red": "red"},
            title=title,
            labels={"truth": "Actual Price", "guess": "Predicted Price"},
            width=800,
            height=600,
        )

        # Assign customdata per trace
        for tr in fig.data:
            mask = df["color"] == tr.name
            tr.customdata = df.loc[mask, ["hover"]].to_numpy()
            tr.hovertemplate = "%{customdata[0]}<extra></extra>"
            tr.marker.update(size=6)

        # Reference line y=x
        fig.add_trace(
            go.Scatter(
                x=[0, max_val],
                y=[0, max_val],
                mode="lines",
                line=dict(width=2, dash="dash", color="deepskyblue"),
                name="y = x",
                hoverinfo="skip",
                showlegend=False,
            )
        )

        fig.update_xaxes(range=[0, max_val])
        fig.update_yaxes(range=[0, max_val])
        fig.update_layout(showlegend=False)
        fig.show()

    def error_trend_chart(self) -> None:
        """Generate error trend chart with confidence intervals"""
        n = len(self.errors)

        # Running mean and std
        running_sums = list(accumulate(self.errors))
        x = list(range(1, n + 1))
        running_means = [s / i for s, i in zip(running_sums, x)]

        running_squares = list(accumulate(e * e for e in self.errors))
        running_stds = [
            math.sqrt((sq_sum / i) - (mean**2)) if i > 1 else 0
            for i, sq_sum, mean in zip(x, running_squares, running_means)
        ]

        # 95% confidence interval
        ci = [1.96 * (sd / math.sqrt(i)) if i > 1 else 0 for i, sd in zip(x, running_stds)]
        upper = [m + c for m, c in zip(running_means, ci)]
        lower = [m - c for m, c in zip(running_means, ci)]

        # Title with final stats
        final_mean = running_means[-1]
        final_ci = ci[-1]
        title = f"{self.title} Error: {final_mean:,.2f} ± {final_ci:,.2f}"

        # Plot
        fig = go.Figure()

        # Shaded confidence interval band
        fig.add_trace(
            go.Scatter(
                x=x + x[::-1],
                y=upper + lower[::-1],
                fill="toself",
                fillcolor="rgba(128,128,128,0.2)",
                line=dict(color="rgba(255,255,255,0)"),
                hoverinfo="skip",
                showlegend=False,
                name="95% CI",
            )
        )

        # Main line
        fig.add_trace(
            go.Scatter(
                x=x,
                y=running_means,
                mode="lines",
                line=dict(width=3, color="firebrick"),
                name="Cumulative Avg Error",
                customdata=list(zip(ci,)),
                hovertemplate=(
                    "n=%{x}<br>"
                    "Avg Error=$%{y:,.2f}<br>"
                    "±95% CI=$%{customdata[0]:,.2f}<extra></extra>"
                ),
            )
        )

        fig.update_layout(
            title=title,
            xaxis_title="Number of Datapoints",
            yaxis_title="Error ($)",
            width=800,
            height=300,
            template="plotly_white",
            showlegend=False,
        )

        fig.show()

    def report(self) -> None:
        """Generate full evaluation report"""
        average_error = sum(self.errors) / self.size
        mse = mean_squared_error(self.truths, self.guesses)
        r2 = r2_score(self.truths, self.guesses) * 100
        
        title = (f"{self.title} results<br>"
                 f"<b>Error:</b> ${average_error:,.2f} "
                 f"<b>MSE:</b> {mse:,.0f} "
                 f"<b>r²:</b> {r2:.1f}%")
        
        self.error_trend_chart()
        self.chart(title)

    def run(self) -> None:
        """Run evaluation and generate report"""
        for i in tqdm(range(self.size)):
            title, guess, truth, error, color = self.run_datapoint(i)
            self.titles.append(title)
            self.guesses.append(guess)
            self.truths.append(truth)
            self.errors.append(error)
            self.colors.append(color)
            print(f"{COLOR_MAP[color]}${error:.0f} ", end="")
        clear_output(wait=True)
        self.report()


def evaluate(function: Callable, data: List[Dict[str, Any]], 
             size: int = DEFAULT_SIZE) -> None:
    """
    Convenience function to evaluate a predictor function
    
    Args:
        function: Predictor function
        data: List of test datapoints
        size: Number of examples to evaluate
    """
    Tester(function, data, size=size).run()


def get_metrics_summary(guesses: List[float], truths: List[float]) -> Dict[str, float]:
    """
    Calculate evaluation metrics
    
    Args:
        guesses: Predicted values
        truths: Ground truth values
        
    Returns:
        Dictionary with metrics
    """
    errors = [abs(g - t) for g, t in zip(guesses, truths)]
    
    return {
        "mean_absolute_error": sum(errors) / len(errors),
        "mse": mean_squared_error(truths, guesses),
        "r2": r2_score(truths, guesses),
        "rmse": math.sqrt(mean_squared_error(truths, guesses)),
    }
