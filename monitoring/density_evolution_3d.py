import multiprocessing
import queue
import logging
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib import cm
from scipy.stats import t, norm

logger = logging.getLogger(__name__)

class DensityEvolutionVisualizer:
    def __init__(self, queue: multiprocessing.Queue):
        self.queue = queue
        self.num_windows = 60  # Nb tranches à afficher
        self.grid_points = 100
        self.x_grid = np.linspace(-0.01, 0.01, self.grid_points) # +/- 1% plage rendements
        
        self.data_history = []
        
        self.fig = None
        self.ax = None
        self.running = True
        self.paused = False

    def _init_plot(self):
        self.fig = plt.figure(figsize=(10, 8))
        self.fig.canvas.manager.set_window_title("View B: Density Evolution (Fat-Tail View)")
        self.ax = self.fig.add_subplot(111, projection='3d')
        
        self.ax.set_xlabel('Returns Space')
        self.ax.set_ylabel('Time (Evolution)')
        self.ax.set_zlabel('Probability Density')
        self.ax.set_title('3D Probability Density Evolution\n(High Kurtosis / Fat Tails Highlighted in Red)', fontweight='bold')
        self.ax.view_init(elev=35, azim=-45)
        
        self.fig.canvas.mpl_connect('key_press_event', self._on_key)

    def _on_key(self, event):
        if event.key == ' ':
            self.paused = not self.paused

    def _update(self, frame):
        if not self.running:
            return
            
        updated = False
        try:
            while True:
                msg = self.queue.get_nowait()
                if msg == "STOP":
                    self.running = False
                    plt.close(self.fig)
                    return
                
                if isinstance(msg, dict):
                    if msg.get('type') == 'UPDATE':
                        data = msg['data']
                        self.data_history.append(data)
                        if len(self.data_history) > self.num_windows:
                            self.data_history.pop(0)
                        updated = True
                
        except queue.Empty:
            pass

        if self.paused or not updated or len(self.data_history) < 2:
            return

        self.ax.clear()
        sampled = self.data_history
        
        # Global x range based on local states
        all_mu = np.array([d.get("mu", 0.0) for d in sampled])
        all_sigma = np.array([d.get("sigma", 0.001) for d in sampled])
        
        x_min = np.min(all_mu - 4 * all_sigma) if len(all_mu) > 0 else -0.01
        x_max = np.max(all_mu + 4 * all_sigma) if len(all_mu) > 0 else 0.01
        if x_min == x_max:
            x_min, x_max = -0.01, 0.01
        x = np.linspace(x_min, x_max, 100)

        for i, d in enumerate(sampled):
            y_time = i
            mu = d.get("mu", np.mean(self.x_grid))
            sigma = max(1e-8, d.get("sigma", 0.001))
            kurt = max(0, d.get("kurtosis", 3.0) - 3.0) # excess kurtosis
            
            # Approximate Fat tails using Student's t distribution
            if kurt > 0.1:
                df = max(2.1, 6.0 / kurt + 4.0)
                scale = sigma * np.sqrt((df - 2.0) / df)
                z = t.pdf(x, df, loc=mu, scale=scale)
            else:
                z = norm.pdf(x, loc=mu, scale=sigma)
                
            # Add small offset for visibility
            z = z + 1e-6
            
            # Fill polygon (Waterfall slice)
            self.ax.plot(x, np.full_like(x, y_time), z, color='midnightblue', linewidth=0.8)
            
            # Color by kurtosis (fat tails)
            color = cm.Reds(min(1.0, kurt / 15.0))
            # Need to format collection explicitly for 3d fill
            poly = Poly3DCollection([list(zip(x, np.full_like(x, y_time), z))], facecolors=color, alpha=0.6)
            self.ax.add_collection3d(poly)

        self.ax.set_xlim(x_min, x_max)
        self.ax.set_ylim(0, self.num_windows)
        self.ax.set_zticks([])

        self.ax.set_xlabel('Returns Space')
        self.ax.set_ylabel('Time (Evolution)')
        self.ax.set_zlabel('Probability Density')
        self.ax.set_title('3D Probability Density Evolution\n(High Kurtosis / Fat Tails Highlighted in Red)', fontweight='bold')
        self.ax.view_init(elev=35, azim=-45)

    def run(self):
        logger.info("Starting Density Evolution Visualizer Process")
        self._init_plot()
        ani = animation.FuncAnimation(self.fig, self._update, interval=200, save_count=50)
        plt.show()

def _start_process(queue):
    viz = DensityEvolutionVisualizer(queue)
    viz.run()

class DensityEvolution3D:
    def __init__(self):
        self.queue = multiprocessing.Queue()
        self.process = None

    def start(self):
        self.process = multiprocessing.Process(target=_start_process, args=(self.queue,))
        self.process.daemon = True
        self.process.start()

    def update(self, pdf_values: np.array, timestamp: float):
        if self.process and self.process.is_alive():
            try:
                self.queue.put({
                    'type': 'UPDATE',
                    'data': {
                        'pdf_values': pdf_values,
                        'timestamp': timestamp
                    }
                })
            except Exception as e:
                logger.error(f"Failed to push to density viz queue: {e}")

    def on_trade(self, data: dict):
        pass

    def stop(self):
        if self.process and self.process.is_alive():
            self.queue.put("STOP")
            self.process.join(timeout=2)
            if self.process.is_alive():
                self.process.terminate()
