
const API_BASE = 'http://localhost:5000/api'

// ═══════════════════════════════════════════
// STATE
// ═══════════════════════════════════════════
let state = {
    hits: [],
    trackAssignments: [],
    numTracks: 0,
    kalmanTrack: null,
    ransacTrack: null
}

// ═══════════════════════════════════════════
// THREE.JS SETUP
// ═══════════════════════════════════════════
let scene, camera, renderer
let hitMeshes = []
let kalmanLine = null
let ransacLine = null

function initThreeJS() {
    const container = document.getElementById('canvas-container')

    // Scene
    scene = new THREE.Scene()
    scene.background = new THREE.Color(0x050508)

    // Camera
    camera = new THREE.PerspectiveCamera(
        60,
        container.clientWidth / container.clientHeight,
        0.1,
        2000
    )
    camera.position.set(100, 80, 100)
    camera.lookAt(0, 0, 0)

    // Renderer
    renderer = new THREE.WebGLRenderer({ antialias: true })
    renderer.setSize(container.clientWidth, container.clientHeight)
    renderer.setPixelRatio(window.devicePixelRatio)
    container.appendChild(renderer.domElement)

    // Lights
    scene.add(new THREE.AmbientLight(0xffffff, 0.6))
    const dirLight = new THREE.DirectionalLight(0xffffff, 0.8)
    dirLight.position.set(50, 100, 50)
    scene.add(dirLight)

    // Grid
    const grid = new THREE.GridHelper(200, 20, 0x1a1a2e, 0x1a1a2e)
    scene.add(grid)

    // Axes helper
    scene.add(new THREE.AxesHelper(20))

    // Mouse controls
    setupMouseControls(container)

    // Resize handler
    window.addEventListener('resize', () => {
        camera.aspect = container.clientWidth / container.clientHeight
        camera.updateProjectionMatrix()
        renderer.setSize(container.clientWidth, container.clientHeight)
    })

    // Animation loop
    function animate() {
        requestAnimationFrame(animate)
        renderer.render(scene, camera)
    }
    animate()
}

// ─── Mouse Controls ───────────────────────
let isDragging = false
let prevMouse = { x: 0, y: 0 }
let spherical = { theta: 0.8, phi: 0.8, radius: 150 }

function setupMouseControls(container) {
    container.addEventListener('mousedown', e => {
        isDragging = true
        prevMouse = { x: e.clientX, y: e.clientY }
    })

    container.addEventListener('mousemove', e => {
        if (!isDragging) return
        const dx = (e.clientX - prevMouse.x) * 0.005
        const dy = (e.clientY - prevMouse.y) * 0.005
        prevMouse = { x: e.clientX, y: e.clientY }
        spherical.theta -= dx
        spherical.phi = Math.max(0.1, Math.min(Math.PI - 0.1, spherical.phi + dy))
        updateCamera()
    })

    container.addEventListener('mouseup', () => isDragging = false)
    container.addEventListener('mouseleave', () => isDragging = false)

    container.addEventListener('wheel', e => {
        spherical.radius *= e.deltaY > 0 ? 1.1 : 0.9
        spherical.radius = Math.max(20, Math.min(500, spherical.radius))
        updateCamera()
    })
}

function updateCamera() {
    camera.position.set(
        spherical.radius * Math.sin(spherical.phi) * Math.sin(spherical.theta),
        spherical.radius * Math.cos(spherical.phi),
        spherical.radius * Math.sin(spherical.phi) * Math.cos(spherical.theta)
    )
    camera.lookAt(0, 0, 0)
}

// ═══════════════════════════════════════════
// VISUALIZATION
// ═══════════════════════════════════════════

const TRACK_COLORS = [0x00ffff, 0xff6600, 0xffff00, 0xff0066, 0x66ff00]

function renderHits(hits, assignments) {
    // Clear old hits
    hitMeshes.forEach(m => scene.remove(m))
    hitMeshes = []

    if (!hits.length) return

    // Calculate center of hits for camera focus
    let centerX = 0, centerY = 0, centerZ = 0
    hits.forEach(h => { centerX += h.x; centerY += h.y; centerZ += h.z })
    centerX /= hits.length
    centerY /= hits.length
    centerZ /= hits.length

    // Create spheres for each hit
    const geo = new THREE.SphereGeometry(0.8, 8, 8)

    hits.forEach((hit, i) => {
        const trackId = assignments[i] || 0
        const color = TRACK_COLORS[trackId % TRACK_COLORS.length]

        const mat = new THREE.MeshPhongMaterial({
            color: color,
            emissive: color,
            emissiveIntensity: 0.5
        })

        const mesh = new THREE.Mesh(geo, mat)
        mesh.position.set(hit.x, hit.y, hit.z)
        scene.add(mesh)
        hitMeshes.push(mesh)
    })

    // Move camera to look at center of hits
    const offset = 80
    spherical.radius = 150
    updateCamera()
    camera.lookAt(centerX, centerY, centerZ)
}

function renderKalmanTrack(data, hits) {
    if (kalmanLine) { scene.remove(kalmanLine); kalmanLine = null }
    if (!data || !hits.length) return

    // Draw line from first to last hit (fitted direction)
    const first = hits[0]
    const last = hits[hits.length - 1]

    const dir = new THREE.Vector3(
        last.x - first.x,
        last.y - first.y,
        last.z - first.z
    ).normalize()

    const startVec = new THREE.Vector3(first.x, first.y, first.z)
    const endVec = new THREE.Vector3(last.x, last.y, last.z)

    // Extend slightly beyond hits
    const extStart = startVec.clone().addScaledVector(dir, -5)
    const extEnd = endVec.clone().addScaledVector(dir, 5)

    const geo = new THREE.BufferGeometry().setFromPoints([extStart, extEnd])
    const mat = new THREE.LineBasicMaterial({ color: 0x00ff88, linewidth: 2 })
    kalmanLine = new THREE.Line(geo, mat)
    scene.add(kalmanLine)
}

function renderRansacTrack(data) {
    if (ransacLine) { scene.remove(ransacLine); ransacLine = null }
    if (!data) return

    const traj = data.trajectory
    const center = new THREE.Vector3(
        traj.start_point.x,
        traj.start_point.y,
        traj.start_point.z
    )
    const dir = new THREE.Vector3(
        traj.direction.x,
        traj.direction.y,
        traj.direction.z
    ).normalize()

    const lineStart = center.clone().addScaledVector(dir, -40)
    const lineEnd = center.clone().addScaledVector(dir, 40)

    const geo = new THREE.BufferGeometry().setFromPoints([lineStart, lineEnd])
    const mat = new THREE.LineBasicMaterial({ color: 0xff00ff, linewidth: 2 })
    ransacLine = new THREE.Line(geo, mat)
    scene.add(ransacLine)
}

function clearScene() {
    hitMeshes.forEach(m => scene.remove(m))
    hitMeshes = []
    if (kalmanLine) { scene.remove(kalmanLine); kalmanLine = null }
    if (ransacLine)  { scene.remove(ransacLine);  ransacLine = null }
}

// ═══════════════════════════════════════════
// API CALLS
// ═══════════════════════════════════════════

async function generateEvent() {
    setStatus('Generating collision event...', 'loading')

    try {
        const numTracks = parseInt(document.getElementById('num-tracks').value)
        const hitsPerTrack = parseInt(document.getElementById('hits-per-track').value)

        const res = await fetch(`${API_BASE}/generate-event`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                num_tracks: numTracks,
                hits_per_track: hitsPerTrack,
                measurement_error: 0.01
            })
        })

        const data = await res.json()
        if (!data.success) throw new Error(data.error)

        // Update state
        state.hits = data.hits
        state.trackAssignments = data.track_assignments
        state.numTracks = data.num_tracks
        state.kalmanTrack = null
        state.ransacTrack = null

        // Render
        renderHits(data.hits, data.track_assignments)

        // Update UI
        document.getElementById('stat-hits').textContent = data.num_hits
        document.getElementById('stat-tracks').textContent = data.num_tracks
        document.getElementById('stat-chi2').textContent = '-'
        document.getElementById('stat-inliers').textContent = '-'
        document.getElementById('btn-kalman').disabled = false
        document.getElementById('btn-ransac').disabled = false
        document.getElementById('kalman-results').style.display = 'none'
        document.getElementById('ransac-results').style.display = 'none'

        setStatus(
            `Generated ${data.num_hits} hits from ${data.num_tracks} particle tracks.`,
            'success'
        )

    } catch (err) {
        setStatus(`Error: ${err.message}. Is Flask server running?`, 'error')
    }
}

async function fitKalman() {
    if (!state.hits.length) return
    setStatus('Fitting Kalman Filter on track 0...', 'loading')

    try {
        // IMPORTANT: Only fit hits from track 0!
        const track0Hits = state.hits.filter(
            (_, i) => state.trackAssignments[i] === 0
        )

        console.log(`Fitting Kalman on ${track0Hits.length} hits from track 0`)

        const res = await fetch(`${API_BASE}/fit-kalman`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ hits: track0Hits })
        })

        const data = await res.json()
        if (!data.success) throw new Error(data.error)

        state.kalmanTrack = data

        // Render
        renderKalmanTrack(data, track0Hits)

        // Update stats
        const chi2 = data.quality.chi2_ndf.toFixed(4)
        document.getElementById('stat-chi2').textContent = chi2
        document.getElementById('kalman-quality').textContent = data.fit_quality
        document.getElementById('kalman-quality').style.color =
            data.fit_quality === 'GOOD' ? '#3fb950' : '#f85149'
        document.getElementById('kalman-chi2').textContent = chi2
        document.getElementById('kalman-hits').textContent = data.quality.hits_used
        document.getElementById('kalman-dir').textContent =
            `(${data.state.px.toFixed(2)}, ${data.state.py.toFixed(2)}, ${data.state.pz.toFixed(2)})`
        document.getElementById('kalman-results').style.display = 'block'

        setStatus(
            `Kalman Filter: ${data.fit_quality} fit. Chi²/NDF = ${chi2}`,
            data.fit_quality === 'GOOD' ? 'success' : 'error'
        )

    } catch (err) {
        setStatus(`Kalman error: ${err.message}`, 'error')
    }
}

async function fitRansac() {
    if (!state.hits.length) return
    setStatus('Running RANSAC on all hits...', 'loading')

    try {
        // RANSAC gets ALL hits - it should find the dominant track!
        const res = await fetch(`${API_BASE}/fit-ransac`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                hits: state.hits,
                fit_type: 'straight'
            })
        })

        const data = await res.json()
        if (!data.success) throw new Error(data.error)

        state.ransacTrack = data

        // Render
        renderRansacTrack(data)

        // Update stats
        const ratio = (data.quality.inlier_ratio * 100).toFixed(1)
        document.getElementById('stat-inliers').textContent = `${ratio}%`
        document.getElementById('ransac-quality').textContent = data.fit_quality
        document.getElementById('ransac-quality').style.color =
            data.fit_quality === 'GOOD' ? '#3fb950' : '#f85149'
        document.getElementById('ransac-inliers').textContent = data.quality.num_inliers
        document.getElementById('ransac-outliers').textContent = data.quality.num_outliers
        document.getElementById('ransac-ratio').textContent = `${ratio}%`
        document.getElementById('ransac-results').style.display = 'block'

        // Physics explanation in status
        const totalHits = state.hits.length
        const inliers = data.quality.num_inliers
        const tracks = state.numTracks
        setStatus(
            `RANSAC found ${inliers}/${totalHits} inliers (${ratio}%). ` +
            `With ${tracks} tracks, ~${(100/tracks).toFixed(0)}% per track is expected!`,
            data.fit_quality === 'GOOD' ? 'success' : 'loading'
        )

    } catch (err) {
        setStatus(`RANSAC error: ${err.message}`, 'error')
    }
}

function clearAll() {
    clearScene()
    state = { hits: [], trackAssignments: [], numTracks: 0, kalmanTrack: null, ransacTrack: null }
    document.getElementById('btn-kalman').disabled = true
    document.getElementById('btn-ransac').disabled = true
    document.getElementById('kalman-results').style.display = 'none'
    document.getElementById('ransac-results').style.display = 'none'
    document.getElementById('stat-hits').textContent = '0'
    document.getElementById('stat-tracks').textContent = '0'
    document.getElementById('stat-chi2').textContent = '-'
    document.getElementById('stat-inliers').textContent = '-'
    setStatus('Cleared. Click "Generate Event" to start.', '')
}

// ═══════════════════════════════════════════
// UI HELPERS
// ═══════════════════════════════════════════

function setStatus(msg, type) {
    const el = document.getElementById('status-message')
    el.textContent = msg
    el.className = `status-message ${type}`
}

// ═══════════════════════════════════════════
// EVENT LISTENERS
// ═══════════════════════════════════════════
document.getElementById('btn-generate').addEventListener('click', generateEvent)
document.getElementById('btn-kalman').addEventListener('click', fitKalman)
document.getElementById('btn-ransac').addEventListener('click', fitRansac)
document.getElementById('btn-clear').addEventListener('click', clearAll)

document.getElementById('num-tracks').addEventListener('input', e => {
    document.getElementById('num-tracks-value').textContent = e.target.value
})
document.getElementById('hits-per-track').addEventListener('input', e => {
    document.getElementById('hits-per-track-value').textContent = e.target.value
})

// ═══════════════════════════════════════════
// INITIALIZE
// ═══════════════════════════════════════════
initThreeJS()
setStatus('Ready. Make sure Flask server is running on port 5000.', '')
