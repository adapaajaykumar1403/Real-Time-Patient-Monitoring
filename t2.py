import random
import numpy as np
import joblib

# Load ML model
try:
    model = joblib.load("model.pkl")
except FileNotFoundError:
    print("Error: model.pkl not found")
    exit()

# Parameter names for consistency
param_names = ["Heart Rate", "Blood Pressure", "Cholesterol", "Oxygen Saturation", "Glucose Level"]

def generate_parameters():
    return {
        "Heart Rate": random.randint(60, 100),
        "Blood Pressure": random.randint(90, 140),
        "Cholesterol": random.randint(150, 250),
        "Oxygen Saturation": round(random.uniform(90, 100), 1),
        "Glucose Level": random.randint(70, 160),
    }

def predict_risk(params):
    values = np.array(list(params.values())).reshape(1, -1)
    return model.predict(values)[0]

# Optional: Tkinter UI (only runs when executing t2.py directly)
if __name__ == "__main__":
    import tkinter as tk
    from tkinter import ttk
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    import matplotlib.pyplot as plt
    from collections import deque

    class RiskMonitorApp:
        def __init__(self, root):
            self.root = root
            self.root.title("Real-Time Patient Parameters")
            self.root.geometry("600x300")

            self.labels = {}
            for idx, param in enumerate(param_names + ["Predicted Risk"]):
                ttk.Label(root, text=param + ":").grid(row=idx, column=0, sticky="w", padx=10, pady=5)
                self.labels[param] = ttk.Label(root, text="--")
                self.labels[param].grid(row=idx, column=1, sticky="w")

            self.show_graph_btn = ttk.Button(root, text="Show Risk Graph", command=self.show_graph_window)
            self.show_graph_btn.grid(row=6, column=0, columnspan=2, pady=10)

            self.x_data = deque(maxlen=50)
            self.y_data = deque(maxlen=50)
            self.time_counter = 0
            self.graph_window = None
            self.graph_canvas = None
            self.graph_fig = None
            self.graph_ax = None

            self.update_data()

        def update_data(self):
            parameters = generate_parameters()
            actual_risk = predict_risk(parameters)

            for key in parameters:
                self.labels[key].config(text=str(parameters[key]))
            self.labels["Predicted Risk"].config(text=str(actual_risk))

            self.x_data.append(self.time_counter)
            self.y_data.append(actual_risk)
            self.time_counter += 1

            if self.graph_ax:
                self.graph_ax.clear()
                self.graph_ax.set_title("Risk Level Over Time")
                self.graph_ax.set_xlabel("Time")
                self.graph_ax.set_ylabel("Risk Level")
                self.graph_ax.set_ylim(0, 5)
                self.graph_ax.set_xlim(max(0, self.time_counter - 50), self.time_counter)

                for i in range(1, len(self.x_data)):
                    self.graph_ax.plot(
                        [self.x_data[i-1], self.x_data[i]],
                        [self.y_data[i-1], self.y_data[i]],
                        color=self.get_line_color(self.y_data[i])
                    )

                self.graph_canvas.draw()

            self.root.after(2000, self.update_data)

        def get_line_color(self, risk):
            if 0 <= risk < 3:
                return 'green'
            elif 3 <= risk < 4:
                return 'orange'
            elif 4 <= risk <= 5:
                return 'red'

        def show_graph_window(self):
            if self.graph_window is None or not self.graph_window.winfo_exists():
                self.graph_window = tk.Toplevel(self.root)
                self.graph_window.title("Real-Time Risk Level Graph")
                self.graph_fig, self.graph_ax = plt.subplots(figsize=(6, 3))
                self.graph_canvas = FigureCanvasTkAgg(self.graph_fig, master=self.graph_window)
                self.graph_canvas.get_tk_widget().pack()
                self.graph_ax.set_title("Risk Level Over Time")
                self.graph_ax.set_xlabel("Time")
                self.graph_ax.set_ylabel("Risk Level")
                self.graph_ax.set_ylim(0, 5)
                self.graph_ax.set_xlim(max(0, self.time_counter - 50), self.time_counter)
                self.graph_canvas.draw()

    root = tk.Tk()
    app = RiskMonitorApp(root)
    root.mainloop()
