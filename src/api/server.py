
import sys
sys.path.insert(0, '/home/estego1965/Documents/PROJECTS/particle-track-reconstruction')

from flask import Flask, jsonify, request
from flask_cors import CORS
import numpy as np
from src.physics.particle_generator import ParticleGenerator
from src.algorithms.kalman_filter import KalmanFilter
from src.algorithms.ransac import RansacTrackFitter

app = Flask(__name__)
CORS(app)

# Global instances
generator = ParticleGenerator()
ransac = RansacTrackFitter(distance_threshold=0.3, iterations=500)


class Hit:
    """Helper class to convert JSON hits to objects"""
    def __init__(self, x, y, z, error_x=0.01, error_y=0.01, error_z=0.01):
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)
        self.error_x = float(error_x)
        self.error_y = float(error_y)
        self.error_z = float(error_z)


@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'OK',
        'message': 'Particle Track Reconstruction Server running'
    })


@app.route('/api/generate-event', methods=['POST'])
def generate_event():
    try:
        data = request.get_json() or {}
        num_tracks = max(1, min(5, data.get('num_tracks', 3)))
        hits_per_track = max(5, min(30, data.get('hits_per_track', 15)))
        measurement_error = max(0.001, min(0.1, data.get('measurement_error', 0.01)))

        generator.clear()
        tracks = generator.generate_multi_track_event(
            num_tracks=num_tracks,
            num_hits_per_track=hits_per_track
        )

        all_hits = []
        track_assignments = []

        for track_id, track_hits in enumerate(tracks):
            for hit in track_hits:
                all_hits.append({
                    'x': float(hit.x),
                    'y': float(hit.y),
                    'z': float(hit.z),
                    'error_x': float(hit.error_x),
                    'error_y': float(hit.error_y),
                    'error_z': float(hit.error_z)
                })
                track_assignments.append(track_id)

        return jsonify({
            'success': True,
            'num_tracks': num_tracks,
            'num_hits': len(all_hits),
            'hits': all_hits,
            'track_assignments': track_assignments
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400


@app.route('/api/fit-kalman', methods=['POST'])
def fit_kalman():
    try:
        data = request.get_json() or {}
        hits_data = data.get('hits', [])

        if not hits_data:
            return jsonify({'success': False, 'error': 'No hits provided'}), 400

        hits = [Hit(h['x'], h['y'], h['z'],
                    h.get('error_x', 0.01),
                    h.get('error_y', 0.01),
                    h.get('error_z', 0.01))
                for h in hits_data]

        kf = KalmanFilter(process_noise=0.001)
        track = kf.fit_track(hits)

        if track is None:
            return jsonify({'success': False, 'error': 'Fitting failed'}), 400

        chi2_ndf = track.chi2 / track.ndf if track.ndf > 0 else 0

        return jsonify({
            'success': True,
            'algorithm': 'Kalman Filter',
            'state': {
                'x':     float(track.state[0]),
                'y':     float(track.state[1]),
                'z':     float(track.state[2]),
                'px':    float(track.state[3]),
                'py':    float(track.state[4]),
                'pz':    float(track.state[5]),
                'q_over_p': float(track.state[6])
            },
            'quality': {
                'chi2':      float(track.chi2),
                'ndf':       int(track.ndf),
                'chi2_ndf':  float(chi2_ndf),
                'hits_used': int(track.hits_used)
            },
            'fit_quality': 'GOOD' if chi2_ndf < 2.0 else 'BAD'
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400


@app.route('/api/fit-ransac', methods=['POST'])
def fit_ransac():
    try:
        data = request.get_json() or {}
        hits_data = data.get('hits', [])
        fit_type = data.get('fit_type', 'straight')

        if not hits_data:
            return jsonify({'success': False, 'error': 'No hits provided'}), 400

        hits = [Hit(h['x'], h['y'], h['z'],
                    h.get('error_x', 0.01),
                    h.get('error_y', 0.01),
                    h.get('error_z', 0.01))
                for h in hits_data]

        if fit_type == 'circle':
            track = ransac.fit_circle(hits)
        else:
            track = ransac.fit_straight_line(hits)

        if track is None:
            return jsonify({'success': False, 'error': 'Fitting failed'}), 400

        inlier_ratio = track.num_inliers / len(hits) if hits else 0

        return jsonify({
            'success': True,
            'algorithm': 'RANSAC',
            'fit_type': fit_type,
            'trajectory': {
                'start_point': {
                    'x': float(track.start_point[0]),
                    'y': float(track.start_point[1]),
                    'z': float(track.start_point[2])
                },
                'direction': {
                    'x': float(track.direction[0]),
                    'y': float(track.direction[1]),
                    'z': float(track.direction[2])
                },
                'curvature': float(track.curvature)
            },
            'quality': {
                'num_inliers':   int(track.num_inliers),
                'num_outliers':  int(track.num_outliers),
                'inlier_ratio':  float(inlier_ratio),
                'total_hits':    len(hits)
            },
            'hit_classification': {
                'inliers':  track.inliers,
                'outliers': track.outliers
            },
            'fit_quality': 'GOOD' if inlier_ratio > 0.8 else 'BAD'
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400


if __name__ == '__main__':
    print("\n" + "="*60)
    print("PARTICLE TRACK RECONSTRUCTION SERVER")
    print("="*60)
    print("Running on: http://localhost:5000")
    print("\nEndpoints:")
    print("  GET  /api/health")
    print("  POST /api/generate-event")
    print("  POST /api/fit-kalman")
    print("  POST /api/fit-ransac")
    print("="*60 + "\n")
    app.run(debug=True, port=5000, host='localhost')
    