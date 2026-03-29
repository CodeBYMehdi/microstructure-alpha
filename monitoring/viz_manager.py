import multiprocessing
import logging
import time
import matplotlib.pyplot as plt
from monitoring.regime_surface_3d import RegimeSurfaceVisualizer
from monitoring.density_evolution_3d import DensityEvolutionVisualizer

logger = logging.getLogger(__name__)

def _run_surface_viz(queue):
    viz = RegimeSurfaceVisualizer(queue)
    viz._init_plot()
    
    # Boucle anim
    anim = import_animation().FuncAnimation(viz.fig, viz._update, interval=100, blit=False, cache_frame_data=False)
    plt.show()

def _run_density_viz(queue):
    viz = DensityEvolutionVisualizer(queue)
    viz._init_plot()

    anim = import_animation().FuncAnimation(viz.fig, viz._update, interval=200, blit=False, cache_frame_data=False)
    plt.show()

def import_animation():
    import matplotlib.animation as animation
    return animation

class VisualizationManager:
    def __init__(self):
        self.q_surface = multiprocessing.Queue()
        self.q_density = multiprocessing.Queue()
        self._p_surface = None
        self._p_density = None
        self._started = False

    def start(self):
        if self._started:
            return
            
        logger.info("Starting 3D Visualizations...")
        
        self._p_surface = multiprocessing.Process(target=_run_surface_viz, args=(self.q_surface,))
        self._p_surface.daemon = True
        self._p_surface.start()
        
        self._p_density = multiprocessing.Process(target=_run_density_viz, args=(self.q_density,))
        self._p_density.daemon = True
        self._p_density.start()
        
        self._started = True

    def stop(self):
        if not self._started:
            return
        
        logger.info("Stopping 3D Visualizations...")
        self.q_surface.put("STOP")
        self.q_density.put("STOP")
        
        if self._p_surface:
            self._p_surface.join(timeout=2)
            if self._p_surface.is_alive(): self._p_surface.terminate()
            
        if self._p_density:
            self._p_density.join(timeout=2)
            if self._p_density.is_alive(): self._p_density.terminate()
            
        self._started = False
