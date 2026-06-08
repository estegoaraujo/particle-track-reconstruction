import numpy as np
from typing import List, Tuple
from dataclasses import dataclass, field


@dataclass
class RansacTrack:
    """Result of RANSAC fitting"""
    inliers: List[int]
    outliers: List[int]
    start_point: np.ndarray
    direction: np.ndarray
    curvature: float = 0.0
    num_inliers: int = 0
    num_outliers: int = 0


class RansacTrackFitter:
    """
    Robust track fitting using RANSAC algorithm
    
    Physics:
    Most hits are from the particle (inliers).
    A few might be noise or cosmic rays (outliers).
    RANSAC finds the trajectory with most votes (inliers).
    """
    
    def __init__(self, distance_threshold: float = 0.3,
                 iterations: int = 500):
        """
        Args:
            distance_threshold: Max distance for hit to be inlier (cm)
            iterations: Number of random samples to try
        """
        self.distance_threshold = distance_threshold
        self.iterations = iterations
    
    def fit_straight_line(self, hits: List) -> RansacTrack:
        """
        Fit straight line through hits using RANSAC
        
        Physics:
        Neutral particle travels in straight line
        (no magnetic force since no charge)
        """
        if len(hits) < 3:
            return None
        
        positions = np.array([[h.x, h.y, h.z] for h in hits])
        num_hits = len(positions)
        
        best_inliers = []
        best_outliers = list(range(num_hits))
        best_point = positions[0]
        best_direction = np.array([1.0, 0.0, 0.0])
        
        for _ in range(self.iterations):
            
            # Step 1: Random sample of 3 hits
            idx = np.random.choice(num_hits, 3, replace=False)
            sample = positions[idx]
            
            # Step 2: Fit line through sample
            p1 = sample[0]
            v = sample[2] - sample[0]
            v_norm = np.linalg.norm(v)
            
            if v_norm < 1e-10:
                continue
            
            direction = v / v_norm
            
            # Step 3: Count inliers
            inliers = []
            outliers = []
            
            for i, pos in enumerate(positions):
                # Distance from point to line
                p_rel = pos - p1
                proj = np.dot(p_rel, direction) * direction
                perp = p_rel - proj
                distance = np.linalg.norm(perp)
                
                if distance <= self.distance_threshold:
                    inliers.append(i)
                else:
                    outliers.append(i)
            
            # Step 4: Keep best model
            if len(inliers) > len(best_inliers):
                best_inliers = inliers
                best_outliers = outliers
                best_point = p1
                best_direction = direction
        
        # Step 5: Refit using ALL inliers
        if len(best_inliers) >= 3:
            inlier_pos = positions[best_inliers]
            center = np.mean(inlier_pos, axis=0)
            
            try:
                centered = inlier_pos - center
                cov = np.cov(centered.T)
                eigenvalues, eigenvectors = np.linalg.eig(cov)
                best_direction = eigenvectors[:, np.argmax(eigenvalues)]
                best_direction = best_direction.real
                best_direction = best_direction / np.linalg.norm(best_direction)
                best_point = center
            except:
                pass
        
        return RansacTrack(
            inliers=best_inliers,
            outliers=best_outliers,
            start_point=best_point,
            direction=best_direction,
            curvature=0.0,
            num_inliers=len(best_inliers),
            num_outliers=len(best_outliers)
        )
    
    def fit_circle(self, hits: List) -> RansacTrack:
        """
        Fit circle (helix projection) through hits using RANSAC
        
        Physics:
        Charged particle curves in magnetic field.
        Projection on xy-plane is circular.
        """
        if len(hits) < 4:
            return None
        
        positions = np.array([[h.x, h.y, h.z] for h in hits])
        num_hits = len(positions)
        
        best_inliers = []
        best_outliers = list(range(num_hits))
        best_center = positions[0][:2]
        best_radius = 1.0
        
        for _ in range(self.iterations):
            
            # Random sample of 4 hits
            idx = np.random.choice(num_hits, 4, replace=False)
            sample = positions[idx, :2]  # xy only
            
            # Fit circle
            result = self._fit_circle_2d(sample)
            if result is None:
                continue
            
            center, radius = result
            
            # Count inliers
            inliers = []
            outliers = []
            
            for i, pos in enumerate(positions):
                xy = pos[:2]
                dist = abs(np.linalg.norm(xy - center) - radius)
                
                if dist <= self.distance_threshold:
                    inliers.append(i)
                else:
                    outliers.append(i)
            
            if len(inliers) > len(best_inliers):
                best_inliers = inliers
                best_outliers = outliers
                best_center = center
                best_radius = radius
        
        if not best_inliers:
            return None
        
        start_point = np.array([
            float(best_center[0]),
            float(best_center[1]),
            float(np.mean(positions[best_inliers, 2]))
        ])
        
        direction = np.array([1.0, 0.0, 0.0])
        
        return RansacTrack(
            inliers=best_inliers,
            outliers=best_outliers,
            start_point=start_point,
            direction=direction,
            curvature=float(best_radius),
            num_inliers=len(best_inliers),
            num_outliers=len(best_outliers)
        )
    
    def _fit_circle_2d(self, points: np.ndarray):
        """Fit circle to 2D points"""
        if len(points) < 3:
            return None
        
        try:
            center = np.mean(points, axis=0)
            distances = np.linalg.norm(points - center, axis=1)
            radius = np.mean(distances)
            
            if radius < 1e-10:
                return None
            
            return (center, radius)
        except:
            return None
    