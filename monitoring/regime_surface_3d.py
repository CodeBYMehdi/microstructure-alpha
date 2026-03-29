import multiprocessing
import queue
import logging
import time
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from matplotlib import cm
import matplotlib.tri as mtri

logger = logging.getLogger(__name__)

class RegimeSurfaceVisualizer:
    def __init__(self, queue: multiprocessing.Queue):
        self.queue = queue
        self.max_points = 200  # Number of points in history for the surface
        self.data_history = [] # List of dicts
        self.fig = None
        self.ax = None
        self.running = True
        self.paused = False

    def _init_plot(self):
        self.fig = plt.figure(figsize=(10, 8))
        self.fig.canvas.manager.set_window_title("View A: Regime Surface (Tracking Volatility vs Drift)")
        self.ax = self.fig.add_subplot(111, projection='3d')
        
        self.ax.set_xlabel('Drift (μ)')
        self.ax.set_ylabel('Volatility (σ)')
        self.ax.set_zlabel('Time Index')
        self.ax.set_title('3D Regime Shift Surface\n(Tracking Volatility vs Drift Over Time)')
        self.ax.view_init(elev=25, azim=135)
        
        self.fig.canvas.mpl_connect('key_press_event', self._on_key)

    def _on_key(self, event):
        if event.key == ' ':
            self.paused = not self.paused
            logger.info(f"Visualization {'paused' if self.paused else 'resumed'}")

    def _update(self, frame):
        if not self.running:
            return
            
        try:
            while True:
                msg = self.queue.get_nowait()
                if msg == "STOP":
                    self.running = False
                    plt.close(self.fig)
                    return
                
                if isinstance(msg, dict):
                    if msg.get('type') == 'UPDATE':
                        self.data_history.append(msg['data'])
                        
        except queue.Empty:
            pass

        if self.paused or len(self.data_history) < 5:
            return

        if len(self.data_history) > self.max_points:
            self.data_history = self.data_history[-self.max_points:]

        drifts = np.array([d['drift'] for d in self.data_history])
        vols = np.array([d['vol'] for d in self.data_history])
        time_idx = np.arange(len(self.data_history))
        
        self.ax.clear()
        self.ax.set_xlabel('Drift (μ)')
        self.ax.set_ylabel('Volatility (σ)')
        self.ax.set_zlabel('Time Index')
        self.ax.set_title('3D Regime Shift Surface\n(Tracking Volatility vs Drift Over Time)', fontweight='bold')
        self.ax.view_init(elev=25, azim=135)
        
        try:
            tri = mtri.Triangulation(drifts, vols)
            surf = self.ax.plot_trisurf(drifts, vols, time_idx, triangles=tri.triangles, cmap='plasma', alpha=0.8, edgecolor='none')
        except Exception:
            surf = self.ax.scatter(drifts, vols, time_idx, c=time_idx, cmap='plasma', s=20)
            
        self.ax.plot(drifts, vols, time_idx, color='black', linewidth=0.8, alpha=0.5)

        # Scale dynamically
        self.ax.set_xlim(np.min(drifts) - 0.1 * abs(np.min(drifts)) - 1e-6, np.max(drifts) + 0.1 * abs(np.max(drifts)) + 1e-6)
        self.ax.set_ylim(0, np.max(vols) * 1.1 + 1e-6)
        self.ax.set_zlim(0, self.max_points)


    def run(self):
        logger.info("Starting Regime Surface Visualizer Process")
        self._init_plot()
        
        ani = animation.FuncAnimation(
            self.fig, 
            self._update, 
            interval=200, 
            blit=False, 
            save_count=50
        )
        
        plt.show()

def _start_process(queue):
    viz = RegimeSurfaceVisualizer(queue)
    viz.run()

class RegimeSurface3D:
    def __init__(self):
        self.queue = multiprocessing.Queue()
        self.process = None

    def start(self):
        self.process = multiprocessing.Process(target=_start_process, args=(self.queue,))
        self.process.daemon = True
        self.process.start()

    def update(self, drift: float, vol: float, density: float, regime_id: int, timestamp: float):
        if self.process and self.process.is_alive():
            try:
                self.queue.put({
                    'type': 'UPDATE',
                    'data': {
                        'drift': drift,
                        'vol': vol,
                        'density': density,
                        'regime_id': regime_id,
                        'timestamp': timestamp
                    }
                })
            except Exception as e:
                logger.error(f"Failed to push to viz queue: {e}")

    def on_trade(self, data: dict):
        pass

    def stop(self):
        if self.process and self.process.is_alive():
            self.queue.put("STOP")
            self.process.join(timeout=2)
            if self.process.is_alive():
                self.process.terminate()
