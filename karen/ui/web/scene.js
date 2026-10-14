// Karen's 3D core: everything here runs on the GPU (Intel Arc), so the
// animation costs almost no CPU. Mode changes ease colors and motion smoothly.
import * as THREE from 'three';
import { EffectComposer } from 'three/addons/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/addons/postprocessing/RenderPass.js';
import { UnrealBloomPass } from 'three/addons/postprocessing/UnrealBloomPass.js';
import { OutputPass } from 'three/addons/postprocessing/OutputPass.js';

const MODES = {
  booting:   { color: 0x4fd8ff, speed: 1.0, distort: 0.10, bloom: 0.7 },
  ready:     { color: 0x4fd8ff, speed: 1.0, distort: 0.09, bloom: 0.62 },
  listening: { color: 0xffb547, speed: 1.35, distort: 0.14, bloom: 0.8 },
  thinking:  { color: 0x9b8cff, speed: 2.3, distort: 0.20, bloom: 0.8 },
  searching: { color: 0x3ddcb0, speed: 2.7, distort: 0.18, bloom: 0.8 },
  speaking:  { color: 0x62e3ff, speed: 1.5, distort: 0.12, bloom: 0.85 },
  error:     { color: 0xff5c7a, speed: 0.8, distort: 0.24, bloom: 0.75 },
};

// Ashima Arts 3D simplex noise (MIT).
const NOISE = /* glsl */`
vec3 mod289(vec3 x){return x-floor(x*(1.0/289.0))*289.0;}
vec4 mod289(vec4 x){return x-floor(x*(1.0/289.0))*289.0;}
vec4 permute(vec4 x){return mod289(((x*34.0)+1.0)*x);}
vec4 taylorInvSqrt(vec4 r){return 1.79284291400159-0.85373472095314*r;}
float snoise(vec3 v){
  const vec2 C=vec2(1.0/6.0,1.0/3.0); const vec4 D=vec4(0.0,0.5,1.0,2.0);
  vec3 i=floor(v+dot(v,C.yyy)); vec3 x0=v-i+dot(i,C.xxx);
  vec3 g=step(x0.yzx,x0.xyz); vec3 l=1.0-g; vec3 i1=min(g.xyz,l.zxy); vec3 i2=max(g.xyz,l.zxy);
  vec3 x1=x0-i1+C.xxx; vec3 x2=x0-i2+C.yyy; vec3 x3=x0-D.yyy; i=mod289(i);
  vec4 p=permute(permute(permute(i.z+vec4(0.0,i1.z,i2.z,1.0))+i.y+vec4(0.0,i1.y,i2.y,1.0))+i.x+vec4(0.0,i1.x,i2.x,1.0));
  float n_=0.142857142857; vec3 ns=n_*D.wyz-D.xzx; vec4 j=p-49.0*floor(p*ns.z*ns.z);
  vec4 x_=floor(j*ns.z); vec4 y_=floor(j-7.0*x_); vec4 x=x_*ns.x+ns.yyyy; vec4 y=y_*ns.x+ns.yyyy;
  vec4 h=1.0-abs(x)-abs(y); vec4 b0=vec4(x.xy,y.xy); vec4 b1=vec4(x.zw,y.zw);
  vec4 s0=floor(b0)*2.0+1.0; vec4 s1=floor(b1)*2.0+1.0; vec4 sh=-step(h,vec4(0.0));
  vec4 a0=b0.xzyw+s0.xzyw*sh.xxyy; vec4 a1=b1.xzyw+s1.xzyw*sh.zzww;
  vec3 p0=vec3(a0.xy,h.x); vec3 p1=vec3(a0.zw,h.y); vec3 p2=vec3(a1.xy,h.z); vec3 p3=vec3(a1.zw,h.w);
  vec4 norm=taylorInvSqrt(vec4(dot(p0,p0),dot(p1,p1),dot(p2,p2),dot(p3,p3)));
  p0*=norm.x; p1*=norm.y; p2*=norm.z; p3*=norm.w;
  vec4 m=max(0.6-vec4(dot(x0,x0),dot(x1,x1),dot(x2,x2),dot(x3,x3)),0.0); m=m*m;
  return 42.0*dot(m*m,vec4(dot(p0,x0),dot(p1,x1),dot(p2,x2),dot(p3,x3)));
}`;

const ease = (t) => 1 - Math.pow(1 - Math.min(Math.max(t, 0), 1), 3);

export class KarenScene {
  constructor(canvas, { reducedMotion = false, idleFps = 24 } = {}) {
    this.canvas = canvas;
    this.idleGap = 1000 / Math.max(5, idleFps) - 1;
    this.reducedMotion = reducedMotion;
    this.mode = 'booting';
    this.level = 0;
    this.targetLevel = 0;
    this.speed = 1;
    this.distort = 0.1;
    this.color = new THREE.Color(MODES.booting.color);
    this.targetColor = this.color.clone();
    this.mouse = new THREE.Vector2();
    this.clock = new THREE.Clock();
    this.spin = 0;
    this.lastFrame = 0;

    const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, powerPreference: 'high-performance' });
    // 80% internal resolution: the glow hides the difference and the GPU does ~36% less work.
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 1.0) * 0.8);
    renderer.setClearColor(0x000000, 0);
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this.renderer = renderer;

    const scene = new THREE.Scene();
    scene.fog = new THREE.FogExp2(0x030a14, 0.035);
    this.scene = scene;
    const camera = new THREE.PerspectiveCamera(42, 1, 0.1, 200);
    camera.position.set(0, 0.5, 22);
    this.camera = camera;

    this.root = new THREE.Group();
    this.root.position.y = 1.15;
    scene.add(this.root);
    this.accentMaterials = [];

    this.#buildCore();
    this.#buildRings();
    this.#buildBars();
    this.#buildParticles();
    this.#buildEnvironment();

    this.composer = new EffectComposer(renderer);
    this.composer.addPass(new RenderPass(scene, camera));
    this.bloom = new UnrealBloomPass(new THREE.Vector2(256, 256), 0.62, 0.42, 0.22);
    this.composer.addPass(this.bloom);
    this.composer.addPass(new OutputPass());

    this.#resize();
    window.addEventListener('resize', () => this.#resize());
    window.addEventListener('pointermove', (event) => {
      this.mouse.set(event.clientX / window.innerWidth - 0.5, event.clientY / window.innerHeight - 0.5);
    });
    this.#applyColor();
  }

  // -- construction -----------------------------------------------------

  #accent(material, strength = 1) {
    material.userData.strength = strength;
    this.accentMaterials.push(material);
    return material;
  }

  #buildCore() {
    this.sphereUniforms = {
      uTime: { value: 0 }, uLevel: { value: 0 }, uDistort: { value: 0.1 },
      uColor: { value: new THREE.Color() }, uReveal: { value: 0 },
    };
    const sphere = new THREE.Mesh(
      new THREE.IcosahedronGeometry(1.15, 28),
      new THREE.ShaderMaterial({
        uniforms: this.sphereUniforms,
        transparent: true, depthWrite: false, blending: THREE.AdditiveBlending,
        vertexShader: NOISE + /* glsl */`
          uniform float uTime, uLevel, uDistort;
          varying vec3 vNormal; varying vec3 vView; varying float vNoise;
          void main() {
            float n = snoise(normal * 1.7 + vec3(uTime * 0.35));
            float n2 = snoise(normal * 4.2 - vec3(uTime * 0.7));
            float d = (n * 0.65 + n2 * 0.25) * (uDistort + uLevel * 0.42);
            vNoise = n;
            vec4 mv = modelViewMatrix * vec4(position + normal * d, 1.0);
            vView = normalize(-mv.xyz);
            vNormal = normalize(normalMatrix * normal);
            gl_Position = projectionMatrix * mv;
          }`,
        fragmentShader: /* glsl */`
          uniform vec3 uColor; uniform float uTime, uLevel, uReveal;
          varying vec3 vNormal; varying vec3 vView; varying float vNoise;
          void main() {
            float fres = pow(1.0 - max(dot(vNormal, vView), 0.0), 2.4);
            float bands = smoothstep(0.86, 1.0, sin(vNoise * 7.0 + uTime * 1.6) * 0.5 + 0.5);
            vec3 col = uColor * (0.05 + fres * 1.05 + bands * 0.28 + uLevel * 0.35);
            col += vec3(1.0) * pow(fres, 8.0) * 0.3;
            gl_FragColor = vec4(col * uReveal, (0.18 + fres * 0.6) * uReveal);
          }`,
      }),
    );
    this.root.add(sphere);
    this.sphere = sphere;

    this.core = new THREE.Mesh(new THREE.SphereGeometry(0.3, 32, 32),
      this.#accent(new THREE.MeshBasicMaterial({ color: 0xffffff }), 1.15));
    this.root.add(this.core);
    this.coreHot = new THREE.Mesh(new THREE.SphereGeometry(0.13, 24, 24), new THREE.MeshBasicMaterial({ color: 0xdff8ff }));
    this.root.add(this.coreHot);

    const shellMaterial = this.#accent(new THREE.LineBasicMaterial({ transparent: true, opacity: 0.22 }), 1);
    this.shell = new THREE.LineSegments(new THREE.EdgesGeometry(new THREE.IcosahedronGeometry(1.72, 1)), shellMaterial);
    this.root.add(this.shell);
    const nodes = new THREE.IcosahedronGeometry(1.72, 1);
    this.nodes = new THREE.Points(nodes, this.#accent(new THREE.PointsMaterial({ size: 0.06, transparent: true, opacity: 0.9 }), 1.4));
    this.shell.add(this.nodes);

    this.satellites = [0, 1, 2].map((index) => {
      const pivot = new THREE.Group();
      pivot.rotation.set(0.6 + index * 0.7, index * 1.3, index * 0.4);
      const dot = new THREE.Mesh(new THREE.SphereGeometry(0.045 + index * 0.01, 12, 12),
        new THREE.MeshBasicMaterial({ color: 0xffffff }));
      dot.position.x = 1.95 + index * 0.28;
      pivot.add(dot);
      this.root.add(pivot);
      return pivot;
    });
  }

  #buildRings() {
    this.gyro = [2.3, 2.62, 2.95].map((radius, index) => {
      const material = this.#accent(new THREE.MeshBasicMaterial({ transparent: true, opacity: 0.55 - index * 0.1 }), 1.1);
      const ring = new THREE.Mesh(new THREE.TorusGeometry(radius, 0.008 + index * 0.003, 6, 240), material);
      ring.rotation.set(Math.PI / 2 + 0.35 * (index - 1), 0.4 * index, 0);
      this.root.add(ring);
      return ring;
    });

    // Segmented reactor ring facing the camera.
    // Repeated pieces are instanced: one draw call each instead of one per piece.
    this.segments = new THREE.Group();
    const segmentMaterial = this.#accent(new THREE.MeshBasicMaterial({ transparent: true, opacity: 0.9 }), 1.3);
    this.segmentCount = 36;
    const arc = (Math.PI * 2) / this.segmentCount;
    this.segmentSets = [0.03, 0.018].map((tube, set) => {
      const indices = [...Array(this.segmentCount).keys()].filter((index) => (index % 3 === 0) === (set === 0));
      const mesh = new THREE.InstancedMesh(new THREE.TorusGeometry(2.08, tube, 4, 10, arc * 0.62), segmentMaterial, indices.length);
      mesh.userData.indices = indices;
      this.segments.add(mesh);
      return mesh;
    });
    this.segmentsSettled = false;
    this.#placeSegments(0);
    this.root.add(this.segments);

    const dashMaterial = this.#accent(new THREE.MeshBasicMaterial({ transparent: true, opacity: 0.35 }), 0.8);
    this.outerDash = new THREE.Group();
    const matrix = new THREE.Matrix4();
    const quaternion = new THREE.Quaternion();
    const zAxis = new THREE.Vector3(0, 0, 1);
    for (const major of [true, false]) {
      const indices = [...Array(120).keys()].filter((index) => (index % 10 === 0) === major);
      const ticks = new THREE.InstancedMesh(new THREE.PlaneGeometry(major ? 0.03 : 0.012, major ? 0.22 : 0.1), dashMaterial, indices.length);
      indices.forEach((index, slot) => {
        const angle = (index / 120) * Math.PI * 2;
        quaternion.setFromAxisAngle(zAxis, angle + Math.PI / 2);
        matrix.compose(new THREE.Vector3(Math.cos(angle) * 3.15, Math.sin(angle) * 3.15, 0), quaternion, new THREE.Vector3(1, 1, 1));
        ticks.setMatrixAt(slot, matrix);
      });
      this.outerDash.add(ticks);
    }
    this.root.add(this.outerDash);
  }

  // Segments assemble one by one during boot; afterwards their matrices never change.
  #placeSegments(t) {
    if (this.segmentsSettled) return;
    const arc = (Math.PI * 2) / this.segmentCount;
    const matrix = new THREE.Matrix4();
    const quaternion = new THREE.Quaternion();
    const zAxis = new THREE.Vector3(0, 0, 1);
    const origin = new THREE.Vector3();
    const scale = new THREE.Vector3();
    let settled = true;
    for (const mesh of this.segmentSets) {
      mesh.userData.indices.forEach((index, slot) => {
        const appear = this.reducedMotion ? 1 : ease((t - 0.6 - index * 0.03) / 0.6);
        if (appear < 1) settled = false;
        quaternion.setFromAxisAngle(zAxis, index * arc);
        scale.setScalar(Math.max(appear, 0.0001));
        matrix.compose(origin, quaternion, scale);
        mesh.setMatrixAt(slot, matrix);
      });
      mesh.instanceMatrix.needsUpdate = true;
    }
    this.segmentsSettled = settled;
  }

  #buildBars() {
    this.barCount = 96;
    const material = this.#accent(new THREE.MeshBasicMaterial({ transparent: true, opacity: 0.85 }), 1.25);
    this.bars = new THREE.InstancedMesh(new THREE.BoxGeometry(0.035, 1, 0.035), material, this.barCount);
    this.bars.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
    this.barMatrix = new THREE.Matrix4();
    this.barQuat = new THREE.Quaternion();
    this.barAxis = new THREE.Vector3(0, 0, 1);
    this.barPos = new THREE.Vector3();
    this.barScale = new THREE.Vector3(1, 1, 1);
    this.root.add(this.bars);
  }

  #buildParticles() {
    const count = 2600;
    const positions = new Float32Array(count * 3);
    const speeds = new Float32Array(count);
    const sizes = new Float32Array(count);
    for (let index = 0; index < count; index += 1) {
      const radius = 2.4 + Math.pow(Math.random(), 1.6) * 4.2;
      const angle = Math.random() * Math.PI * 2;
      positions[index * 3] = Math.cos(angle) * radius;
      positions[index * 3 + 1] = (Math.random() - 0.5) * 0.35 * (radius / 3);
      positions[index * 3 + 2] = Math.sin(angle) * radius;
      speeds[index] = (0.08 + Math.random() * 0.22) * (3.2 / radius);
      sizes[index] = Math.random() * 2.2 + 0.6;
    }
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    geometry.setAttribute('aSpeed', new THREE.BufferAttribute(speeds, 1));
    geometry.setAttribute('aSize', new THREE.BufferAttribute(sizes, 1));
    this.particleUniforms = { uTime: { value: 0 }, uColor: { value: new THREE.Color() }, uReveal: { value: 0 }, uPixel: { value: 1 } };
    const material = new THREE.ShaderMaterial({
      uniforms: this.particleUniforms, transparent: true, depthWrite: false, blending: THREE.AdditiveBlending,
      vertexShader: /* glsl */`
        uniform float uTime, uPixel; attribute float aSpeed; attribute float aSize; varying float vFade;
        void main() {
          float a = uTime * aSpeed;
          vec3 p = vec3(position.x * cos(a) - position.z * sin(a), position.y, position.x * sin(a) + position.z * cos(a));
          vec4 mv = modelViewMatrix * vec4(p, 1.0);
          vFade = smoothstep(7.0, 2.4, length(position.xz));
          gl_PointSize = aSize * uPixel * (7.0 / -mv.z);
          gl_Position = projectionMatrix * mv;
        }`,
      fragmentShader: /* glsl */`
        uniform vec3 uColor; uniform float uReveal; varying float vFade;
        void main() {
          float d = length(gl_PointCoord - 0.5);
          float alpha = smoothstep(0.5, 0.0, d) * vFade * uReveal;
          gl_FragColor = vec4(uColor * 1.4, alpha);
        }`,
    });
    this.disk = new THREE.Points(geometry, material);
    this.disk.rotation.set(1.18, 0, 0.22);
    this.root.add(this.disk);
  }

  #buildEnvironment() {
    const stars = new Float32Array(1400 * 3);
    for (let index = 0; index < 1400; index += 1) {
      const radius = 40 + Math.random() * 60;
      const theta = Math.random() * Math.PI * 2;
      const phi = Math.acos(2 * Math.random() - 1);
      stars[index * 3] = radius * Math.sin(phi) * Math.cos(theta);
      stars[index * 3 + 1] = radius * Math.sin(phi) * Math.sin(theta);
      stars[index * 3 + 2] = radius * Math.cos(phi);
    }
    const starGeometry = new THREE.BufferGeometry();
    starGeometry.setAttribute('position', new THREE.BufferAttribute(stars, 3));
    this.stars = new THREE.Points(starGeometry, new THREE.PointsMaterial({ color: 0x6f9fc4, size: 0.12, transparent: true, opacity: 0.55, fog: false }));
    this.scene.add(this.stars);

    this.floor = new THREE.PolarGridHelper(9, 24, 10, 96);
    this.floor.material.transparent = true;
    this.floor.material.opacity = 0.16;
    this.floor.material.depthWrite = false;
    this.#accent(this.floor.material, 0.7);
    this.floor.position.y = -3.6;
    this.scene.add(this.floor);

    const pad = new THREE.Mesh(new THREE.RingGeometry(1.2, 2.6, 96, 1),
      this.#accent(new THREE.MeshBasicMaterial({ transparent: true, opacity: 0.04, side: THREE.DoubleSide, depthWrite: false, blending: THREE.AdditiveBlending }), 0.8));
    pad.rotation.x = -Math.PI / 2;
    pad.position.y = -3.58;
    this.pad = pad;
    this.scene.add(pad);

    // Light beam from the floor pad up to the core.
    const beam = new THREE.Mesh(new THREE.CylinderGeometry(0.9, 1.6, 3.2, 48, 1, true),
      this.#accent(new THREE.MeshBasicMaterial({ transparent: true, opacity: 0.018, side: THREE.DoubleSide, depthWrite: false, blending: THREE.AdditiveBlending }), 1));
    beam.position.y = -1.6;
    this.scene.add(beam);
  }

  // -- control ------------------------------------------------------------

  setMode(mode) {
    this.mode = MODES[mode] ? mode : 'ready';
    this.targetColor = new THREE.Color(MODES[this.mode].color);
  }

  setLevel(value) { this.targetLevel = Math.max(0, Math.min(1, value)); }

  get active() { return ['booting', 'listening', 'thinking', 'searching', 'speaking'].includes(this.mode); }

  #resize() {
    const width = window.innerWidth;
    const height = window.innerHeight;
    this.renderer.setSize(width, height, false);
    this.composer.setSize(width, height);
    this.bloom.resolution.set(width / 2, height / 2);
    this.particleUniforms.uPixel.value = this.renderer.getPixelRatio();
    this.camera.aspect = width / height;
    // Keep the core comfortably framed on narrow windows.
    this.camera.fov = width / height < 1.4 ? 50 : 42;
    this.camera.updateProjectionMatrix();
  }

  #applyColor() {
    for (const material of this.accentMaterials) {
      material.color.copy(this.color).multiplyScalar(material.userData.strength || 1);
    }
    this.sphereUniforms.uColor.value.copy(this.color);
    this.particleUniforms.uColor.value.copy(this.color);
  }

  start() {
    const loop = (now) => {
      requestAnimationFrame(loop);
      // 60 fps while active, 24 fps when idle, 20 fps with reduced motion.
      const minGap = this.reducedMotion ? 50 : this.active || this.level > 0.02 ? 0 : this.idleGap;
      if (now - this.lastFrame < minGap) return;
      this.lastFrame = now;
      this.#frame();
    };
    requestAnimationFrame(loop);
  }

  #frame() {
    const dt = Math.min(this.clock.getDelta(), 0.1);
    const t = this.clock.elapsedTime;
    const params = MODES[this.mode];
    const motion = this.reducedMotion ? 0.25 : 1;
    const reveal = this.reducedMotion ? 1 : ease(t / 2.6);

    // Smoothly ease color, speed and distortion toward the current mode.
    const k = 1 - Math.exp(-dt * 4);
    if (!this.color.equals(this.targetColor)) {
      this.color.lerp(this.targetColor, k);
      this.#applyColor();
    }
    this.speed += (params.speed - this.speed) * k;
    this.distort += (params.distort - this.distort) * k;
    this.bloom.strength += (params.bloom - this.bloom.strength) * k;

    let target = this.targetLevel;
    if (this.mode === 'speaking') {
      target = 0.32 + 0.38 * Math.abs(Math.sin(t * 7.3) * Math.sin(t * 2.9 + 1.2)) + Math.random() * 0.1;
    } else if (this.mode === 'thinking' || this.mode === 'searching') {
      target = 0.16 + 0.06 * Math.sin(t * 3);
    }
    this.level += (target - this.level) * (1 - Math.exp(-dt * 12));
    this.spin += dt * this.speed * motion;

    // Camera: cinematic dolly-in during boot, then gentle drift and mouse parallax.
    const camZ = 22 - 8.2 * ease(t / 2.8);
    this.camera.position.x += ((this.mouse.x * 1.6 + Math.sin(t * 0.15) * 0.4) - this.camera.position.x) * 0.04;
    this.camera.position.y += ((0.7 - this.mouse.y * 1.0 + Math.cos(t * 0.12) * 0.2) - this.camera.position.y) * 0.04;
    this.camera.position.z = camZ;
    this.camera.lookAt(0, 0.2, 0);

    this.sphereUniforms.uTime.value = t * (0.6 + this.speed * 0.4);
    this.sphereUniforms.uLevel.value = this.level;
    this.sphereUniforms.uDistort.value = this.distort;
    this.sphereUniforms.uReveal.value = reveal;
    this.particleUniforms.uTime.value = this.spin * 0.9;
    this.particleUniforms.uReveal.value = reveal;

    const breathe = 1 + Math.sin(t * 1.7) * 0.04;
    const coreScale = (breathe + this.level * 0.7) * reveal;
    this.core.scale.setScalar(Math.max(coreScale, 0.001));
    this.coreHot.scale.setScalar(Math.max(coreScale, 0.001));
    this.sphere.rotation.y = this.spin * 0.25;
    this.shell.rotation.set(this.spin * 0.12, this.spin * 0.2, 0);
    this.shell.scale.setScalar(Math.max(reveal * (1 + this.level * 0.12), 0.001));

    this.gyro.forEach((ring, index) => {
      ring.rotation.z += dt * this.speed * motion * (0.3 + index * 0.18) * (index % 2 ? -1 : 1);
      ring.rotation.x += dt * this.speed * motion * 0.05 * (index - 1);
      ring.scale.setScalar(Math.max(ease((t - 0.3 - index * 0.25) / 1.6), 0.001));
    });
    this.segments.rotation.z = -this.spin * 0.35;
    this.#placeSegments(t);
    this.outerDash.rotation.z = this.spin * 0.08;
    this.outerDash.scale.setScalar(Math.max(ease((t - 1.0) / 1.4), 0.001));
    this.satellites.forEach((pivot, index) => { pivot.rotation.y += dt * this.speed * motion * (0.9 + index * 0.45); });
    this.disk.rotation.z = 0.22 + Math.sin(t * 0.2) * 0.05;
    this.stars.rotation.y = t * 0.006;
    this.floor.rotation.y = -this.spin * 0.05;
    this.pad.material.opacity = 0.03 + this.level * 0.06;

    // Audio-reactive bars around the reactor ring.
    const barRadius = 2.42;
    for (let index = 0; index < this.barCount; index += 1) {
      const angle = (index / this.barCount) * Math.PI * 2 + this.spin * 0.1;
      let amp;
      if (this.mode === 'thinking' || this.mode === 'searching') {
        amp = 0.04 + Math.pow((Math.sin(t * 5 - index * 0.35) + 1) / 2, 3) * 0.45;
      } else {
        const jitter = (Math.sin(t * 9 + index * 1.7) + Math.sin(t * 5.3 + index * 0.9)) * 0.25 + 0.5;
        amp = 0.03 + jitter * this.level * 1.15;
      }
      amp *= reveal;
      const length = Math.max(amp, 0.002);
      this.barPos.set(Math.cos(angle) * (barRadius + length / 2), Math.sin(angle) * (barRadius + length / 2), 0);
      this.barQuat.setFromAxisAngle(this.barAxis, angle - Math.PI / 2);
      this.barScale.set(1, length, 1);
      this.barMatrix.compose(this.barPos, this.barQuat, this.barScale);
      this.bars.setMatrixAt(index, this.barMatrix);
    }
    this.bars.instanceMatrix.needsUpdate = true;

    this.composer.render(dt);
  }
}
