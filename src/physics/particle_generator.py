
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple

@dataclass
class DetectorHit:
    x: float              # Position in x (cm)
    y: float              # Position in y (cm)
    z: float              # Position in z (cm)
    error_x: float = 0.1  # Measurement uncertainty x (mm)
    error_y: float = 0.1  # Measurement uncertainty y (mm)
    error_z: float = 0.1  # Measurement uncertainty z (mm)
    time: float = 0.0     # Time of detection
    detector_layer: int = 0  # Which layer of detector

class ParticleGenerator: 
    
    def __init__(self, seed: int = None):
    
        if seed is not None:
            np.random.seed(seed)
        self.hits: List[DetectorHit] = []
    
    def generate_straight_track(
        self,
        start_pos: np.ndarray,
        direction: np.ndarray,
        num_hits: int = 10,
        spacing: float = 1.0,
        measurement_error: float = 0.1
    ) -> List[DetectorHit]:
    
        # Normalize direction to unit vector (length = 1)
        direction = direction / np.linalg.norm(direction)
        hits = []

        for i in range(num_hits):
            # True position (exact, no noise)
            t = i * spacing  # Distance along trajectory
            true_pos = start_pos + t * direction
            
            # Add measurement noise (realistic detector error)
            # Physics: Normal distribution (Gaussian noise)
            # Why? Thermal noise, quantum effects, electronics
            measured_pos = true_pos + np.random.normal(
                0,  # Mean (centered at true position)
                measurement_error,  # Standard deviation (0.1 mm)
                3  # 3D (x, y, z)
            )
            
            hit = DetectorHit(
                x=measured_pos[0],
                y=measured_pos[1],
                z=measured_pos[2],
                error_x=measurement_error,
                error_y=measurement_error,
                error_z=measurement_error,
                detector_layer=i
            )
            hits.append(hit)
        
        self.hits.extend(hits)
        return hits
    
    def generate_curved_track(
        self,
        start_pos: np.ndarray,
        initial_direction: np.ndarray,
        curvature: float,
        num_hits: int = 20,
        measurement_error: float = 0.1
    ) -> List[DetectorHit]:
       
        hits = []
        
        # Helix parameters
        angle_step = 0.2 / curvature if curvature > 0 else 0
        radius = curvature
        
        for i in range(num_hits):
            angle = i * angle_step
            
            # Helical trajectory (circular + linear)
            # x, y: circular motion
            x = start_pos[0] + radius * np.sin(angle)
            y = start_pos[1] + radius * (1 - np.cos(angle))
            # z: linear motion
            z = start_pos[2] + i * 0.5
            
            # Add measurement noise
            measured_pos = np.array([x, y, z]) + np.random.normal(
                0, measurement_error, 3
            )
            
            hit = DetectorHit(
                x=measured_pos[0],
                y=measured_pos[1],
                z=measured_pos[2],
                error_x=measurement_error,
                error_y=measurement_error,
                error_z=measurement_error,
                detector_layer=i
            )
            hits.append(hit)
        
        self.hits.extend(hits)
        return hits
    
    def generate_multi_track_event(
        self,
        num_tracks: int = 2,
        num_hits_per_track: int = 15
    ) -> List[List[DetectorHit]]:
        
        event_tracks = []
        
        for track_id in range(num_tracks):
            # Random starting position in detector volume
            # Physics: Particles are created at collision point,
            # then spread out in different directions
            start_pos = np.random.uniform(-50, 50, 3)
            
            # Random direction (isotropic = equally likely in all directions)
            # Physics: Collision is quantum process, direction is random
            direction = np.random.uniform(-1, 1, 3)
            direction /= np.linalg.norm(direction)
            
            # Random curvature (50% straight, 50% curved)
            # Physics: 50% neutrals (straight), 50% charged (curved)
            curvature = np.random.uniform(5, 50) if np.random.random() > 0.5 else 0
            
            if curvature > 0:
                # Charged particle (curved track)
                track_hits = self.generate_curved_track(
                    start_pos, direction, curvature, num_hits_per_track
                )
            else:
                # Neutral particle (straight track)
                track_hits = self.generate_straight_track(
                    start_pos, direction, num_hits_per_track
                )
            
            event_tracks.append(track_hits)
        
        return event_tracks
    
    def clear(self):
        self.hits.clear()