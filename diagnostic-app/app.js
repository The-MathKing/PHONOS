// --- THREE.JS VISUALIZER SETUP ---
const container = document.getElementById('canvas-container');
const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
renderer.setSize(window.innerWidth, window.innerHeight);
container.appendChild(renderer.domElement);

// Create a parametric topological mesh to represent the vocal tract/submental surface
const geometry = new THREE.PlaneGeometry(10, 10, 32, 32);
const material = new THREE.MeshBasicMaterial({ 
    color: 0x00ffcc, 
    wireframe: true,
    transparent: true,
    opacity: 0.8
});
const mesh = new THREE.Mesh(geometry, material);
mesh.rotation.x = -Math.PI / 2;
scene.add(mesh);

camera.position.z = 12;
camera.position.y = 5;
camera.lookAt(0, 0, 0);

// Create Emotional Aura (Particle System)
const particleGeometry = new THREE.BufferGeometry();
const particleCount = 1000;
const particlePositions = new Float32Array(particleCount * 3);
for (let i = 0; i < particleCount * 3; i++) {
    particlePositions[i] = (Math.random() - 0.5) * 15;
}
particleGeometry.setAttribute('position', new THREE.BufferAttribute(particlePositions, 3));
const particleMaterial = new THREE.PointsMaterial({
    color: 0xffffff,
    size: 0.1,
    transparent: true,
    opacity: 0.5
});
const particles = new THREE.Points(particleGeometry, particleMaterial);
scene.add(particles);

// Global target states for smooth interpolation
let targetState = {
    pitch: 0, yaw: 0, intensity: 0, f1: 0, f2: 0, f3: 0, arousal: 0, valence: 0
};

// Animation Loop
function animate() {
    requestAnimationFrame(animate);

    // Smoothly interpolate current mesh state towards target state
    mesh.rotation.z += (targetState.yaw * 0.5 - mesh.rotation.z) * 0.1;
    mesh.scale.y += (1.0 + targetState.pitch * 0.5 - mesh.scale.y) * 0.1;
    
    // Modulate intensity via opacity
    material.opacity += (0.2 + targetState.intensity * 0.8 - material.opacity) * 0.1;

    // Distort vertices based on formants to visualize spatial morphology
    const positions = geometry.attributes.position;
    const time = Date.now() * 0.001;
    
    for(let i = 0; i < positions.count; i++) {
        const x = positions.getX(i);
        const y = positions.getY(i);
        
        // Z displacement driven by f1, f2, f3 interacting as standing waves
        let z = Math.sin(x * 2.0 + time * 5.0) * targetState.f1 * 1.5;
        z += Math.cos(y * 2.0 + time * 4.0) * targetState.f2 * 1.5;
        z += Math.sin((x+y) * 1.5 + time * 6.0) * targetState.f3 * 1.5;
        
        positions.setZ(i, z);
    }
    positions.needsUpdate = true;

    // Modulate Emotional Aura (Particles)
    // Arousal drives speed/turbulence
    particles.rotation.y += 0.001 + targetState.arousal * 0.05;
    particles.rotation.x += 0.001 + targetState.arousal * 0.03;
    
    // Valence drives color (Negative=Red, Positive=Blue/Green)
    const normalizedValence = (targetState.valence + 1) / 2; // 0 to 1
    const r = 1.0 - normalizedValence;
    const g = normalizedValence;
    const b = normalizedValence;
    particleMaterial.color.setRGB(r, g, b);
    particleMaterial.opacity = 0.2 + targetState.arousal * 0.8;

    renderer.render(scene, camera);
}
animate();

window.addEventListener('resize', () => {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
});


// --- WEBSOCKET API IMPLEMENTATION (Node.js Relay) ---
const socket = io("http://localhost:3000");

socket.on('connect', () => {
    console.log("Connected to Node.js Relay Server");
    // Optionally update UI connection status
});

socket.on('telemetry', (data) => {
    if (data.type === 'ml_state') {
        targetState.pitch = data.pitch;
        targetState.yaw = data.yaw;
        targetState.intensity = data.intensity;
        targetState.f1 = data.f1;
        targetState.f2 = data.f2;
        targetState.f3 = data.f3;
        targetState.arousal = data.arousal;
        targetState.valence = data.valence;

        // Update UI text
        document.getElementById('val-pitch').innerText = targetState.pitch.toFixed(3);
        document.getElementById('val-yaw').innerText = targetState.yaw.toFixed(3);
        document.getElementById('val-intensity').innerText = targetState.intensity.toFixed(3);
        document.getElementById('val-f1').innerText = targetState.f1.toFixed(3);
        document.getElementById('val-f2').innerText = targetState.f2.toFixed(3);
        document.getElementById('val-f3').innerText = targetState.f3.toFixed(3);
        document.getElementById('val-arousal').innerText = targetState.arousal.toFixed(3);
        document.getElementById('val-valence').innerText = targetState.valence.toFixed(3);
    } else if (data.type === 'raw_dsp') {
        // Phase B Raw Output Handling (Could be used to drive an oscilloscope visualization in the future)
        // console.log("Raw DSP: ", data.a0, data.a1, data.a2, data.a3);
    }
});
