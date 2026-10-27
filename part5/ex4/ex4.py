#! /usr/bin/env python3

# this is now exercise 2 and 3


import json
import numpy as np
import matplotlib.pyplot as plt
import sys
import argparse
import os
from scipy.optimize import least_squares

def signal_handler(sig, frame):
    """Graceful termination on Ctrl+C."""
    print("\n[INFO] Exiting application...")
    sys.exit(0)


def parse_arguments():
    """Sets up CLI flags and returns them as a dictionary."""
    ap = argparse.ArgumentParser(description="SAVI Optimization Pipeline")
    ap.add_argument("-c", "--capture", action="store_true", help="Launch interactive point collection")
    ap.add_argument("-f", "--file", type=str, default="points.json", help="Path to JSON dataset")
    ap.add_argument("-ni", "--iterations", type=int, default=1000, help="Number of random iterations")
    ap.add_argument("-m", "--model", type=str, default="line", choices=["line", "composed"], help="Model type")

    return vars(ap.parse_args())

class InteractiveLineFitter:
    """
    Two-stage estimator:
    1. Collects points interactively from user clicks via Matplotlib GUI.
    2. Performs stochastic line fitting (y = m * x + b) against the captured points.
    """

    def __init__(self, x_lim=(-10, 10), y_lim=(-5, 5)):
        self.x_lim = x_lim
        self.y_lim = y_lim

        self.xs = []
        self.ys = []
        self.best_params = (0.0, 0.0)
        self.best_error = float('inf')

    # ==========================================
    # STAGE 1: POINT SELECTION (MATPLOTLIB GUI)
    # ==========================================
    def capture_points(self, output_json: str = "points.json"):
        """Opens an interactive canvas for mouse-click point placement until ESC is pressed."""
        fig, ax = plt.subplots()
        ax.set_xlim(self.x_lim)
        ax.set_ylim(self.y_lim)
        ax.set_title("Click to add points. Press ESC when done.")
        ax.grid(True)

        (scatter_plot,) = ax.plot([], [], 'ro')

        def on_click(event):
            if event.inaxes != ax:
                return
            self.xs.append(float(event.xdata))
            self.ys.append(float(event.ydata))
            scatter_plot.set_data(self.xs, self.ys)
            fig.canvas.draw_idle()

        def on_key(event):
            if event.key == 'escape':
                plt.close(fig)

        fig.canvas.mpl_connect('button_press_event', on_click)
        fig.canvas.mpl_connect('key_press_event', on_key)

        plt.show()  # Blocks until ESC closes the window

        # Export collected data
        data = {'xs': self.xs, 'ys': self.ys}
        with open(output_json, "w") as f:
            json.dump(data, f, indent=4)
        print(f"Captured {len(self.xs)} points and saved to '{output_json}'.")

    # ==========================================
    # STAGE 2: STOCHASTIC FITTING
    # ==========================================
    @staticmethod
    def predict(x: np.ndarray, m: float, b: float) -> np.ndarray:
        return m * x + b

    def compute_error(self, m: float, b: float) -> float:
        xs_arr = np.array(self.xs)
        ys_arr = np.array(self.ys)
        ys_pred = self.predict(xs_arr, m, b)
        return float(np.sum(np.abs(ys_pred - ys_arr)))

    def fit_and_animate(self, num_iterations: int = 100, m_range=(-3.0, 3.0), b_range=(-5.0, 5.0)):
        """Searches for parameters (m, b) that minimize L1 error with live rendering."""
        if len(self.xs) < 2:
            raise ValueError("Select at least 2 points before fitting.")

        plt.ion()
        fig, ax = plt.subplots()
        ax.plot(self.xs, self.ys, 'ro', label='Ground Truth')
        ax.set_xlim(self.x_lim)
        ax.set_ylim(self.y_lim)
        ax.grid(True)

        x_eval = np.linspace(self.x_lim[0], self.x_lim[1], 200)
        (line_plot,) = ax.plot([], [], 'b-', label='Candidate Model')
        ax.legend()

        for step in range(num_iterations):
            m_cand = np.random.uniform(*m_range)
            b_cand = np.random.uniform(*b_range)

            error = self.compute_error(m_cand, b_cand)

            if error < self.best_error:
                self.best_error = error
                self.best_params = (m_cand, b_cand)

            # Draw current candidate line
            line_plot.set_data(x_eval, self.predict(x_eval, m_cand, b_cand))
            ax.set_title(f"Iter {step + 1}/{num_iterations} | Err: {error:.2f} | Best: {self.best_error:.2f}")
            plt.pause(0.04)

        plt.ioff()

        # Display final best fit line
        m_best, b_best = self.best_params
        line_plot.set_data(x_eval, self.predict(x_eval, m_best, b_best))
        ax.set_title(f"Finished: y = {m_best:.2f}x + {b_best:.2f} | Min Err: {self.best_error:.2f}")
        plt.show()

    def fit_least_squares(self):
        # 1. THE RESIDUAL WORKHORSE
        def model_residuals(params, x_eval, y_true):
            """
            Computes the signed residual vector r_i = y_pred - y_true.
            Notice: returns an ARRAY of length n, NOT a sum!
            """
            m_param, b_param = params

            # --- GENERIC PLACEHOLDER: Compute line prediction ---
            y_pred = m_param * x_eval + b_param

            # --- Return raw 1D error vector ---
            return y_pred - y_true

        # 2. CONVERT INTERNAL STATE TO NUMPY ARRAYS
        # (Replace these placeholders with your actual internal coordinates)
        x_data = np.array(self.xs, dtype=float)
        y_data = np.array(self.ys, dtype=float)

        # 3. INITIAL SEED (Starting point on the error hill)
        # What is a reasonable neutral guess for [slope, intercept]?
        initial_guess = [1, 0]

        # 4. DISPATCH THE SOLVER
        # args passes (x_data, y_data) directly into model_residuals
        optimization_output = least_squares(
            model_residuals,
            x0=initial_guess,
            args=(x_data, y_data)
        )

        # 5. HARVEST THE OPTIMAL PARAMETERS
        # The fitted array lives inside the '.x' attribute of the result object
        optimal_parameters = optimization_output.x  # contains [m_opt, b_opt]

        # Save or return your result
        return optimal_parameters

    def plot_line(self, params):
        m_opt, b_opt = params

        fig, ax = plt.subplots()
        ax.plot(self.xs, self.ys, 'ro', label='Ground Truth')
        ax.set_xlim(self.x_lim)
        ax.set_ylim(self.y_lim)
        ax.grid(True)

        x_eval = np.linspace(self.x_lim[0], self.x_lim[1], 200)
        y_eval = self.predict(x_eval, m_opt, b_opt)
        ax.plot(x_eval, y_eval, 'g-', linewidth=2, label=f'SciPy Fit (y = {m_opt:.2f}x + {b_opt:.2f})')

        ax.set_title("SciPy Least Squares Fit")
        ax.legend()
        plt.show()

    def load_points(self, json_filepath: str = "points.json"):
        """Loads points from a JSON file, checking if it exists first."""
        if not os.path.exists(json_filepath):
            raise FileNotFoundError(f"File '{json_filepath}' not found!")

        with open(json_filepath, "r") as f:
            data = json.load(f)
        self.xs = data['xs']
        self.ys = data['ys']
        print(f"Loaded {len(self.xs)} points from '{json_filepath}'.")

    # 1. FUNÇÃO DE PREDIÇÃO VETORIZADA
    @staticmethod
    def predict_composed(x, m, b1, a, c, b2):
        """Calcula y para qualquer array x com base nos dois ramos."""
        return np.where(x <= 0, m * x + b1, a * (x ** 2) + c * x + b2)

    # 2. PARTE A: OTIMIZAÇÃO ALEATÓRIA (RANDOM FITTING)
    def fit_composed_random(self):
        # Read the object's internal data:
        x_data = np.array(self.xs, dtype=float)
        y_data = np.array(self.ys, dtype=float)

        def residuals(params, x_eval, y_true):
            # Evaluate using your static method:
            y_pred = self.predict_composed(x_eval, *params)
            return y_pred - y_true

        initial_guess = [1.0, 1.0, 1.0, 1.0, 1.0]
        res = least_squares(residuals, x0=initial_guess, args=(x_data, y_data))
        return res.x

    def plot_composed(self, params):
        fig, ax = plt.subplots()
        ax.plot(self.xs, self.ys, 'ro', label='Ground Truth')
        ax.axvline(0, color='gray', linestyle=':', label='x = 0')  # Domain boundary

        x_eval = np.linspace(self.x_lim[0], self.x_lim[1], 400)
        # Unpack the 5 parameters (*params) into the prediction function:
        y_eval = self.predict_composed(x_eval, *params)

        ax.plot(x_eval, y_eval, 'g-', label='Composed Fit')
        ax.legend()
        plt.show()

    # 3. PARTE B: OTIMIZAÇÃO POR SCI-PY (LEAST SQUARES)
    def fit_composed_least_squares(xs, ys):
        def residuals(params, x_eval, y_true):
            # Desempacota os 5 coeficientes
            m, b1, a, c, b2 = params
            y_pred = predict_composed(x_eval, m, b1, a, c, b2)
            # Retorna o vetor 1D de diferenças
            return y_pred - y_true

        # Chute inicial com 5 variáveis [m0, b1_0, a0, c0, b2_0]
        initial_guess = [0.0, 0.0, 0.0, 0.0, 0.0]

        resultado = least_squares(residuals, x0=initial_guess, args=(xs, ys))
        return resultado.x  # Retorna [m, b1, a, c, b2] ótimos



# --- Execution ---
if __name__ == "__main__":
    args = parse_arguments() # getting the args from argparser
    app = InteractiveLineFitter(x_lim=(-10, 10), y_lim=(-5, 5))

    # Decision: Capture fresh points or read existing file
    if args['capture']:
        app.capture_points(args['file'])
    else:
        app.load_points(args['file'])

    # Decision: Model routing
    if args['model'] == "composed":
        # Call your composed fitting methods using args['iterations']
        pass

    # Step 1: Click points on the plot, then press ESC
    # app.capture_points("points.json")

    # Step 2: Fit model to the captured points
    # app.fit_and_animate(num_iterations=120)


    m_opt, b_opt = app.fit_least_squares()
    print(f"SciPy Solution: y = {m_opt:.3f}x + {b_opt:.3f}")

    params = app.fit_least_squares()
    app.plot_line(params)