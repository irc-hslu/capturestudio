"use client";
import {useCallback, useEffect, useMemo, useRef, useState} from "react";
import * as THREE from "three";
import {OrbitControls} from "three/examples/jsm/controls/OrbitControls.js";

/* ---------------- types ---------------- */
type CameraInfo = {
    id: string;
    index: string;
    position: [number, number, number];                 // OpenCV world
    rotation: [number, number, number, number];         // WXYZ quaternion (c2w in OpenCV)
    intrinsics: { fx: number; fy: number; cx: number; cy: number; w: number; h: number };
};
type Manifest = {
    sessionId: string;
    title: string;
    performers: string[];
    location: string;
    date: string;
    time: string;
    fps: number;
    pointBudget: number;
    cameras: CameraInfo[];
    reconstructions?: Array<{
        frameStart: number; totalFrames: number;
        depthSources: Record<string, { label: string; dir: string }>;
        recon_types: string[]; defaultDepthSource?: string;
    }>;
};
type Props = {
    session: string;
    start: number;
    total: number;
    recon: "pcd" | "gs";
    depth: string;
    modality: "color" | "depth" | "normals" | string;
    manifest: Manifest;
};

type WorkerFrame = {
    positions: Float32Array;   // OpenCV world coords (raw)
    colors: Uint8Array;        // RGB 0..255
    normals?: Float32Array;
    count: number;
};

/* ---------------- tunables ---------------- */
const TARGET_FPS = 30;
const MAX_INFLIGHT = 1;
const PREFETCH_AHEAD = 4;
const MANIFEST_IS_C2W = true;   // set false if a session is w2c
const UNIT_SCALE = 1.0;         // 0.001 if positions are in mm
const POINT_SIZE = 0.02;
const GT_NEAR = 0.08;
const GT_FRUSTUM_VIS_SCALE = 0.6;
const BG = 0x0B0B0B; // background color (black)

const SWITCH_HZ = 10;           // check nearest cam 10x/sec
const SWITCH_HYSTERESIS = 0.15; // 15% closer required to switch

/* ---------------- OpenCV -> OpenGL transforms ---------------- */
const S3 = new THREE.Matrix3().set(1, 0, 0, 0, -1, 0, 0, 0, -1);
const S_apply_vec = (v: THREE.Vector3) => new THREE.Vector3(v.x, -v.y, -v.z);
const mat3FromQuat = (q: THREE.Quaternion) =>
    new THREE.Matrix3().setFromMatrix4(new THREE.Matrix4().makeRotationFromQuaternion(q));
const toOpenGL_R_c2w = (Rcv: THREE.Matrix3) => {
    const SR = new THREE.Matrix3().multiplyMatrices(S3, Rcv);
    return new THREE.Matrix3().multiplyMatrices(SR, S3); // S * R * S
};
const quatWXYZ_to_THREE = (wxyz: [number, number, number, number]) => {
    const [w, x, y, z] = wxyz;
    return new THREE.Quaternion(x, y, z, w); // THREE expects (x,y,z,w)
};
const quatFromMat3 = (R: THREE.Matrix3) => {
    const X = new THREE.Vector3(1, 0, 0).applyMatrix3(R).normalize();
    const Y = new THREE.Vector3(0, 1, 0).applyMatrix3(R).normalize();
    const Z = new THREE.Vector3(0, 0, 1).applyMatrix3(R).normalize();
    const m4 = new THREE.Matrix4().makeBasis(X, Y, Z);
    return new THREE.Quaternion().setFromRotationMatrix(m4);
};

/* ---------------- pose conversion (grounded) ---------------- */
function poseFromManifestC2W_OpenCV_to_Three(
    rot_wxyz: [number, number, number, number],
    t_cv: [number, number, number],
    manifestIsC2W: boolean
) {
    const q_cv = quatWXYZ_to_THREE(rot_wxyz);
    let R_cv = mat3FromQuat(q_cv);               // camera->world (OpenCV)
    if (!manifestIsC2W) R_cv = new THREE.Matrix3().copy(R_cv).transpose(); // w2c -> c2w

    const R_gl = toOpenGL_R_c2w(R_cv);
    const t_gl = S_apply_vec(new THREE.Vector3(...t_cv)).multiplyScalar(UNIT_SCALE);
    const q_gl = quatFromMat3(R_gl);
    return {position: t_gl, quaternion: q_gl};
}

/* ---------------- frustum builder ---------------- */
function makeFrustumForCam(cam: CameraInfo, manifestIsC2W: boolean): THREE.LineSegments {
    const {position, quaternion} = poseFromManifestC2W_OpenCV_to_Three(cam.rotation, cam.position, manifestIsC2W);
    const {w, h, fx, fy} = cam.intrinsics;

    const near = GT_NEAR;
    const widthNear = (near * w) / fx;
    const heightNear = (near * h) / fy;

    const fwd = new THREE.Vector3(0, 0, -1).applyQuaternion(quaternion).normalize();
    const right = new THREE.Vector3(1, 0, 0).applyQuaternion(quaternion).normalize();
    const up = new THREE.Vector3(0, 1, 0).applyQuaternion(quaternion).normalize();

    const centerNear = position.clone().addScaledVector(fwd, near);
    const hx = (widthNear * GT_FRUSTUM_VIS_SCALE) / 2;
    const hy = (heightNear * GT_FRUSTUM_VIS_SCALE) / 2;

    const p00 = centerNear.clone().addScaledVector(right, -hx).addScaledVector(up, +hy);
    const p10 = centerNear.clone().addScaledVector(right, +hx).addScaledVector(up, +hy);
    const p11 = centerNear.clone().addScaledVector(right, +hx).addScaledVector(up, -hy);
    const p01 = centerNear.clone().addScaledVector(right, -hx).addScaledVector(up, -hy);

    const verts = new Float32Array([
        // edges
        position.x, position.y, position.z, p00.x, p00.y, p00.z,
        position.x, position.y, position.z, p10.x, p10.y, p10.z,
        position.x, position.y, position.z, p11.x, p11.y, p11.z,
        position.x, position.y, position.z, p01.x, p01.y, p01.z,
        // base
        p00.x, p00.y, p00.z, p10.x, p10.y, p10.z,
        p10.x, p10.y, p10.z, p11.x, p11.y, p11.z,
        p11.x, p11.y, p11.z, p01.x, p01.y, p01.z,
        p01.x, p01.y, p01.z, p00.x, p00.y, p00.z,
    ]);
    const geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.BufferAttribute(verts, 3));
    const fr = new THREE.LineSegments(geo, new THREE.LineBasicMaterial({color: 0x888888}));
    fr.name = `frustum:${cam.id}`;
    return fr;
}

/* ---------------- worker ---------------- */
function makeWorker(): Worker {
    try {
        return new Worker(new URL("../workers/plyWorker.ts", import.meta.url), {type: "module"});
    } catch {
        return new Worker("/workers/plyWorker.js", {type: "module"});
    }
}

/* ========================================================================== */

export default function ThreeViewer(p: Props) {
    const mountRef = useRef<HTMLDivElement>(null);
    const rendererRef = useRef<THREE.WebGLRenderer | null>(null);
    const cameraRef = useRef<THREE.PerspectiveCamera | null>(null);
    const controlsRef = useRef<OrbitControls | null>(null);
    const sceneRef = useRef<THREE.Scene | null>(null);

    // points
    const ptsRef = useRef<THREE.Points | null>(null);
    const posAttrRef = useRef<THREE.BufferAttribute | null>(null);
    const colAttrRef = useRef<THREE.BufferAttribute | null>(null);

    // frusta
    const frustaGroupRef = useRef<THREE.Group | null>(null);
    const meanVisRef = useRef<THREE.Group | null>(null);
    const gtCamPoseRef = useRef<Map<string, { pos: THREE.Vector3; quat: THREE.Quaternion }>>(new Map());

    // state
    const [camsOnDisk, setCamsOnDisk] = useState<string[]>([]);
    const [currentCamId, setCurrentCamId] = useState<string | null>(null);
    const [tsNames, setTsNames] = useState<string[]>([]);
    const [status, setStatus] = useState<string>("");

    // playback/cache
    const playIdxRef = useRef(0);
    const isPlayingRef = useRef(true);
    const [isPlaying, setIsPlaying] = useState(true);
    const timerRef = useRef<number | null>(null);

    const workerRef = useRef<Worker | null>(null);
    const resolversRef = useRef<Map<string, (ok: boolean) => void>>(new Map());
    const cacheRef = useRef<Map<string, WorkerFrame>>(new Map());
    const inflightRef = useRef<Set<string>>(new Set());

    const centroidRef = useRef<THREE.Vector3 | null>(null);
    const radiusRef = useRef<number>(1);
// which axis we treat as "vertical on screen" for yaw-only
// diagnostics (already have mean vectors earlier)
    const meanUpRef = useRef<THREE.Vector3>(new THREE.Vector3(0, 1, 0));
    const meanFwdRef = useRef<THREE.Vector3>(new THREE.Vector3(0, 0, -1));
    const meanRightRef = useRef<THREE.Vector3>(new THREE.Vector3(1, 0, 0));// which axis to yaw around (we’ll set this to mean-up once computed)

// ---- yaw clamping state ----
    const yawAxisRef = useRef(new THREE.Vector3(0, 1, 0)); // mean-up
    const yawCenterRef = useRef(new THREE.Vector3());        // orbit pivot (center)
    const yawURef = useRef(new THREE.Vector3(1, 0, 0)); // basis in yaw plane (reference forward)
    const yawVRef = useRef(new THREE.Vector3(0, 0, 1)); // second basis = yawAxis x yawU
    const yawAngleRef = useRef(0);                          // current yaw angle (rad)
    const yawMinRef = useRef(-Infinity);                  // allowed angle range
    const yawMaxRef = useRef(Infinity);

// angle of a direction (projected into yaw plane) in the {U,V} basis
    function dirToYawAngle(dir: THREE.Vector3) {
        const a = yawAxisRef.current;
        const u = yawURef.current;
        const v = yawVRef.current;
        const proj = dir.clone().sub(a.clone().multiplyScalar(dir.dot(a)));
        if (proj.lengthSq() === 0) return 0;
        proj.normalize();
        const x = proj.dot(u);
        const y = proj.dot(v);
        return Math.atan2(y, x); // [-pi, pi]
    }

// put camera on the yaw circle at a given angle (keep current distance)
    function placeCameraAtAngle(angle: number) {
        const cam = cameraRef.current, ctr = controlsRef.current;
        if (!cam || !ctr) return;
        const target = ctr.target.clone();
        const dist = cam.position.distanceTo(target);

        // rel = rotate( -dist * U ) around yawAxis by 'angle'
        const rel0 = yawURef.current.clone().multiplyScalar(-dist);
        const q = new THREE.Quaternion().setFromAxisAngle(yawAxisRef.current, angle);
        const rel = rel0.applyQuaternion(q);

        cam.position.copy(target.clone().add(rel));
        cam.up.copy(yawAxisRef.current);
        cam.lookAt(target);
        cam.updateMatrixWorld();
        ctr.update();

        yawAngleRef.current = angle;
    }

// rotate camera around target by 'angle' radians around 'axis'
    function orbitAroundAxis(axis: THREE.Vector3, angle: number) {
        const cam = cameraRef.current, ctr = controlsRef.current;
        if (!cam || !ctr) return;
        const target = ctr.target.clone();
        const rel = cam.position.clone().sub(target);

        const q = new THREE.Quaternion().setFromAxisAngle(axis.clone().normalize(), angle);
        rel.applyQuaternion(q);

        cam.position.copy(target.clone().add(rel));
        cam.up.copy(axis);       // keep the camera’s up locked to your yaw axis
        cam.lookAt(target);
        cam.updateMatrixWorld();

        // tell OrbitControls its target stayed the same; no internal spherical rotation
        ctr.update();
    }

// attach a yaw-only drag on the renderer canvas; keep wheel zoom via OrbitControls
    function attachYawOnlyMouseDrag() {
        const r = rendererRef.current, ctr = controlsRef.current;
        if (!r || !ctr) return;

        // Disable OrbitControls' own rotation; we will drive yaw ourselves.
        ctr.enableRotate = false;
        ctr.enablePan = false;
        ctr.enableZoom = true;
        ctr.enableDamping = false; // not needed when we directly place the camera

        let dragging = false;
        let lastX = 0;

        const onPointerDown = (e: PointerEvent) => {
            if (e.button !== 0) return; // left button only
            dragging = true;
            lastX = e.clientX;
            (r.domElement as HTMLElement).style.cursor = "ew-resize";
            r.domElement.setPointerCapture(e.pointerId);
        };

        const onPointerMove = (e: PointerEvent) => {
            if (!dragging) return;
            const dx = e.clientX - lastX;
            lastX = e.clientX;

            // // scale drag pixels -> radians (tweak 0.003 as you like)
            // const angle = dx * 0.003;

            // dx -> radians (tweak sensitivity)
            const angleDelta = dx * 0.003;
            // proposed new angle
            let next = yawAngleRef.current + angleDelta;
            // clamp to [yawMin, yawMax]
            next = Math.max(yawMinRef.current, Math.min(yawMaxRef.current, next));
            placeCameraAtAngle(next);

            // orbitAroundAxis(yawAxisRef.current, angle);
        };

        const onPointerUp = (e: PointerEvent) => {
            if (!dragging) return;
            dragging = false;
            (r.domElement as HTMLElement).style.cursor = "default";
            r.domElement.releasePointerCapture(e.pointerId);
        };

        // Touch (one finger = yaw)
        const onTouchStart = (e: TouchEvent) => {
            if (e.touches.length !== 1) return;
            dragging = true;
            lastX = e.touches[0].clientX;
            (r.domElement as HTMLElement).style.cursor = "ew-resize";
        };

        const onTouchMove = (e: TouchEvent) => {
            if (!dragging || e.touches.length !== 1) return;
            const dx = e.touches[0].clientX - lastX;
            lastX = e.touches[0].clientX;
            // const angle = dx * 0.003;

            const angleDelta = dx * 0.003;
            // proposed new angle
            let next = yawAngleRef.current + angleDelta;
            // clamp to [yawMin, yawMax]
            next = Math.max(yawMinRef.current, Math.min(yawMaxRef.current, next));
            placeCameraAtAngle(next);
            // orbitAroundAxis(yawAxisRef.current, angle);
        };

        const onTouchEnd = () => {
            dragging = false;
            (r.domElement as HTMLElement).style.cursor = "default";
        };

        // Store handlers on the element so we can remove them later if needed
        r.domElement.addEventListener("pointerdown", onPointerDown);
        r.domElement.addEventListener("pointermove", onPointerMove);
        r.domElement.addEventListener("pointerup", onPointerUp);
        r.domElement.addEventListener("touchstart", onTouchStart, {passive: true});
        r.domElement.addEventListener("touchmove", onTouchMove, {passive: true});
        r.domElement.addEventListener("touchend", onTouchEnd);

        // simple cleanup hook
        return () => {
            r.domElement.removeEventListener("pointerdown", onPointerDown);
            r.domElement.removeEventListener("pointermove", onPointerMove);
            r.domElement.removeEventListener("pointerup", onPointerUp);
            r.domElement.removeEventListener("touchstart", onTouchStart);
            r.domElement.removeEventListener("touchmove", onTouchMove);
            r.domElement.removeEventListener("touchend", onTouchEnd);
        };
    }

    // --- Orientation quick-test knobs ---
    const TWIST_DEG = 0; // try: 0, 90, 180, -90
    const TWIST_AXIS: "up" | "right" | "forward" = "up"; // which axis to rotate around

    function projectOntoPlane(v: THREE.Vector3, n: THREE.Vector3) {
        return v.clone().sub(n.clone().multiplyScalar(v.dot(n)));
    }

    function pickHorizontalForward(desiredFwd: THREE.Vector3, up: THREE.Vector3) {
        let f = projectOntoPlane(desiredFwd.clone().normalize(), up);
        if (f.lengthSq() < 1e-10) {
            const tmp = Math.abs(up.y) < 0.9 ? new THREE.Vector3(0, 1, 0) : new THREE.Vector3(1, 0, 0);
            f = projectOntoPlane(tmp, up);
        }
        return f.normalize();
    }

    /** Place camera so: up=yawAxis, forward≈forwardHorizontal, lookAt(target) */
    function placeCameraYawLook(
        cam: THREE.PerspectiveCamera,
        target: THREE.Vector3,
        yawAxis: THREE.Vector3,
        desiredFwd: THREE.Vector3,
        distance: number
    ) {
        const up = yawAxis.clone().normalize();
        const f = pickHorizontalForward(desiredFwd, up);      // remove roll (purely horizontal fwd)
        const pos = target.clone().addScaledVector(f, -distance);
        cam.up.copy(up);
        cam.position.copy(pos);
        cam.lookAt(target);
    }

    /** Lock OrbitControls to yaw-only around yawAxis, centered at target */
    function lockOrbitYawOnly(yawAxis: THREE.Vector3, target: THREE.Vector3) {
        const cam = cameraRef.current, ctr = controlsRef.current;
        if (!cam || !ctr) return;

        const up = yawAxis.clone().normalize();
        cam.up.copy(up);
        ctr.target.copy(target);

        // Clamp polar angle (pitch) to current value w.r.t. chosen up
        const rel = cam.position.clone().sub(target).normalize();
        const phi = Math.acos(THREE.MathUtils.clamp(rel.dot(up), -1, 1)); // angle from up
        ctr.minPolarAngle = phi;
        ctr.maxPolarAngle = phi;

        ctr.enablePan = false;
        ctr.enableRotate = true;  // yaw only (polar clamped)
        ctr.enableZoom = true;

        ctr.update();
        ctr.saveState();
    }

    /** Lock OrbitControls to yaw (azimuth) only around `yawAxis`, looking at `target`. */
    function lockOrbitToAxis(yawAxis: THREE.Vector3, target: THREE.Vector3) {
        const cam = cameraRef.current, ctr = controlsRef.current;
        if (!cam || !ctr) return;

        const up = yawAxis.clone().normalize();
        cam.up.copy(up);
        ctr.target.copy(target);

        // lock polar angle (pitch) to whatever it is now relative to the chosen up-axis
        const rel = cam.position.clone().sub(target);
        const rnorm = rel.clone().normalize();
        const phi = Math.acos(THREE.MathUtils.clamp(rnorm.dot(up), -1, 1)); // angle to up
        ctr.minPolarAngle = phi;
        ctr.maxPolarAngle = phi;

        ctr.enablePan = false;
        ctr.enableRotate = true;  // yaw only (since polar is clamped)
        ctr.enableZoom = true;

        ctr.update();
        ctr.saveState();
    }

    // continuity across camera switches
    const lastTsRef = useRef<string | null>(null);

    /* ----- root dir derived from manifest entry ----- */
    const rootDir = useMemo(() => {
        const entry = (p.manifest.reconstructions || []).find(r => r.frameStart === p.start && r.totalFrames === p.total);
        const dir = entry?.depthSources?.[p.depth]?.dir;
        return dir && dir.startsWith(p.recon + "_") ? dir : `${p.recon}_${p.depth}`;
    }, [p.manifest, p.start, p.total, p.recon, p.depth]);

    /* ----- helper: place + sync orbit controls (anti-snap) ----- */
    function placeAndSyncCamera(
        position: THREE.Vector3,
        quaternion: THREE.Quaternion,
        target: THREE.Vector3
    ) {
        const cam = cameraRef.current!;
        const ctr = controlsRef.current!;
        if (!cam || !ctr) return;

        cam.up.set(0, 1, 0);
        cam.position.copy(position);
        cam.quaternion.copy(quaternion);

        ctr.target.copy(target);
        ctr.enableDamping = true;
        ctr.dampingFactor = 0.08;
        ctr.screenSpacePanning = false;
        ctr.minPolarAngle = THREE.MathUtils.degToRad(5);
        ctr.maxPolarAngle = THREE.MathUtils.degToRad(175);
        ctr.minDistance = 0.15;
        ctr.maxDistance = 3.0;
        ctr.zoomSpeed = 0.8;
        ctr.rotateSpeed = 0.9;

        ctr.update();   // <-- critical to avoid first-wheel snap
        ctr.saveState();
    }

    useEffect(() => {
        // force recompute of subject pivot when we switch recon/depth/cam
        centroidRef.current = null;
        radiusRef.current = 1;
    }, [rootDir, currentCamId]);

    /* ----- three init ----- */
    /* ----- three init ----- */
    useEffect(() => {
        if (!mountRef.current) return;

        // Scene
        const scene = new THREE.Scene();
        scene.background = new THREE.Color(BG);
        sceneRef.current = scene;

        // Renderer
        const renderer = new THREE.WebGLRenderer({antialias: true});
        renderer.setPixelRatio(window.devicePixelRatio);
        renderer.setSize(mountRef.current.clientWidth, mountRef.current.clientHeight);
        renderer.setClearColor(BG, 1);
        renderer.setClearAlpha(1);
        mountRef.current.appendChild(renderer.domElement);
        rendererRef.current = renderer;

        // Camera + Controls
        const camera = new THREE.PerspectiveCamera(
            50,
            mountRef.current.clientWidth / mountRef.current.clientHeight,
            0.01,
            5000
        );
        cameraRef.current = camera;

        const controls = new OrbitControls(camera, renderer.domElement);
        controls.enablePan = false;
        controlsRef.current = controls;

        // Attach yaw-only drag AFTER controls exist
        const detachYaw = attachYawOnlyMouseDrag();

        // Prealloc points buffer
        const budget = p.manifest.pointBudget || 300_000;
        const geom = new THREE.BufferGeometry();
        const posA = new THREE.BufferAttribute(new Float32Array(budget * 3), 3);
        const colA = new THREE.Uint8BufferAttribute(new Uint8Array(budget * 3), 3, true);
        posA.setUsage(THREE.DynamicDrawUsage);
        colA.setUsage(THREE.DynamicDrawUsage);
        geom.setAttribute("position", posA);
        geom.setAttribute("color", colA);
        geom.setDrawRange(0, 0);
        posAttrRef.current = posA;
        colAttrRef.current = colA;

        const ptsMat = new THREE.PointsMaterial({
            size: POINT_SIZE,
            vertexColors: true,
            sizeAttenuation: true,
        });
        const pts = new THREE.Points(geom, ptsMat);
        pts.frustumCulled = false;
        scene.add(pts);
        ptsRef.current = pts;

        // Frusta container
        const frusta = new THREE.Group();
        frusta.name = "gt-frustums";
        frustaGroupRef.current = frusta;
        scene.add(frusta);

        // Worker
        const worker = makeWorker();
        workerRef.current = worker;
        const onMsg = (ev: MessageEvent) => {
            const d = ev.data;
            const url: string | undefined = d?.url;
            if (!url) return;

            inflightRef.current.delete(url);
            const resolve = resolversRef.current.get(url);
            resolversRef.current.delete(url);

            if (d.ok === false) {
                resolve?.(false);
                return;
            }
            const frame: WorkerFrame = {
                positions: new Float32Array(d.positions),
                colors: new Uint8Array(d.colors),
                normals: d.normals ? new Float32Array(d.normals) : undefined,
                count: d.count,
            };
            cacheRef.current.set(url, frame);
            resolve?.(true);
        };
        worker.addEventListener("message", onMsg);

        // Resize
        const onResize = () => {
            if (!rendererRef.current || !cameraRef.current || !mountRef.current) return;
            cameraRef.current.aspect = mountRef.current.clientWidth / mountRef.current.clientHeight;
            cameraRef.current.updateProjectionMatrix();
            rendererRef.current.setSize(mountRef.current.clientWidth, mountRef.current.clientHeight);
        };
        window.addEventListener("resize", onResize);

        // RAF loop
        let rafId = 0;
        const loop = () => {
            // Keep the orbit target welded to centroid when available
            const ctr = controlsRef.current;
            if (ctr && centroidRef.current) ctr.target.copy(centroidRef.current);

            ctr?.update(); // once per frame is enough
            renderer.render(scene, camera);
            rafId = requestAnimationFrame(loop);
        };
        rafId = requestAnimationFrame(loop);

        // Cleanup
        return () => {
            cancelAnimationFrame(rafId);
            window.removeEventListener("resize", onResize);

            worker.removeEventListener("message", onMsg);
            worker.terminate();

            detachYaw?.();

            // Dispose GPU resources
            pts.geometry.dispose();
            (pts.material as THREE.Material).dispose();

            controls.dispose();
            renderer.dispose();

            if (renderer.domElement.parentElement) {
                renderer.domElement.parentElement.removeChild(renderer.domElement);
            }

            // Clear refs (optional but nice with fast refresh)
            sceneRef.current = null;
            rendererRef.current = null;
            cameraRef.current = null;
            controlsRef.current = null;
            ptsRef.current = null;
            posAttrRef.current = null;
            colAttrRef.current = null;
            frustaGroupRef.current = null;
        };
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, []);


    /* ----- build GT frustums from manifest + cache their poses (and set initial view) ----- */
    useEffect(() => {
        if (!sceneRef.current || !frustaGroupRef.current) return;
        const group = frustaGroupRef.current;
        for (const child of [...group.children]) group.remove(child);

        const present = new Set(camsOnDisk.length ? camsOnDisk : p.manifest.cameras.map(c => c.id));
        const cams = p.manifest.cameras.filter(c => present.has(c.id));
        if (!cams.length) return;

        gtCamPoseRef.current.clear();
        for (const cam of cams) {
            const fr = makeFrustumForCam(cam, MANIFEST_IS_C2W);
            group.add(fr);

            const {position, quaternion} =
                poseFromManifestC2W_OpenCV_to_Three(cam.rotation, cam.position, MANIFEST_IS_C2W);
            gtCamPoseRef.current.set(cam.id, {pos: position.clone(), quat: quaternion.clone()});
        }

        // ---- mean up/forward diagnostics (uses the same grounded conversion you already use)
        type Pose = { position: THREE.Vector3; quaternion: THREE.Quaternion };
        const poses: Pose[] = cams.map(c => {
            const {position, quaternion} =
                poseFromManifestC2W_OpenCV_to_Three(c.rotation, c.position, MANIFEST_IS_C2W);
            return {position, quaternion};
        });

        if (poses.length) {
            // average camera-space axes in world coords
            const meanUp = new THREE.Vector3();
            const meanFwd = new THREE.Vector3();
            const meanRight = new THREE.Vector3();
            const center = new THREE.Vector3();

            for (const pz of poses) {
                meanUp.add(new THREE.Vector3(0, 1, 0).applyQuaternion(pz.quaternion).normalize());
                meanFwd.add(new THREE.Vector3(0, 0, -1).applyQuaternion(pz.quaternion).normalize());
                meanRight.add(new THREE.Vector3(1, 0, 0).applyQuaternion(pz.quaternion).normalize());
                center.add(pz.position);
            }
            meanUp.normalize();
            meanFwd.normalize();
            meanRight.normalize();
            center.multiplyScalar(1 / poses.length);

            // save mean axes
            meanUpRef.current.copy(meanUp);
            meanFwdRef.current.copy(meanFwd);
            meanRightRef.current.copy(meanRight);

// --- lock yaw axis & center ---
            yawAxisRef.current.copy(meanUp).normalize();
            yawCenterRef.current.copy(center);

// reference forward in yaw plane (no roll)
            const forwardNoRoll = meanFwd.clone().sub(meanUp.clone().multiplyScalar(meanFwd.dot(meanUp))).normalize();
            yawURef.current.copy(forwardNoRoll);
            yawVRef.current.copy(new THREE.Vector3().crossVectors(yawAxisRef.current, yawURef.current).normalize());

// --- angular window from first/last cameras ---
            const firstCam = cams[0];
            const lastCam = cams[cams.length - 1];

            const pFirst = poseFromManifestC2W_OpenCV_to_Three(firstCam.rotation, firstCam.position, MANIFEST_IS_C2W).position;
            const pLast = poseFromManifestC2W_OpenCV_to_Three(lastCam.rotation, lastCam.position, MANIFEST_IS_C2W).position;

            const dirFirst = pFirst.clone().sub(center).normalize();
            const dirLast = pLast.clone().sub(center).normalize();

            let aFirst = dirToYawAngle(dirFirst);
            let aLast = dirToYawAngle(dirLast);

// unwrap so the span is the smaller arc
// ensure aLast >= aFirst
            if (aLast < aFirst) [aFirst, aLast] = [aLast, aFirst];
            if (aLast - aFirst > Math.PI) { // take the other arc
                                            // shift both by +2π so span < π
                aFirst += 2 * Math.PI;
            }

// small margin (optional)
            const M = THREE.MathUtils.degToRad(4);
            yawMinRef.current = Math.min(aFirst, aLast) - M;
            yawMaxRef.current = Math.max(aFirst, aLast) + M;

// set initial yaw angle = clamp(0) i.e., pointing along yawU (your current good start)
            const a0 = THREE.MathUtils.clamp(0, yawMinRef.current, yawMaxRef.current);
            yawAngleRef.current = a0;

// (Re)place the camera to match these parameters (keeps distance)
            placeCameraAtAngle(a0);

// ---- distance limits from camera ring ----
            const dists = cams.map(c => {
                const pos = poseFromManifestC2W_OpenCV_to_Three(c.rotation, c.position, MANIFEST_IS_C2W).position;
                return pos.distanceTo(center);
            });
            const minD = Math.max(0.15, Math.min(...dists) * 0.75);
            const maxD = Math.max(minD + 0.01, Math.max(...dists) * 1.30);

            if (controlsRef.current) {
                controlsRef.current.minDistance = minD;
                controlsRef.current.maxDistance = maxD;
                controlsRef.current.update();
            }

// orbit/yaw axis is mean up
            const yawAxis = meanUpRef.current.clone().normalize();
            yawAxisRef.current.copy(yawAxis);

// pick twist axis based on the knob
            const twistAxis =
                TWIST_AXIS === "up" ? meanUpRef.current :
                    TWIST_AXIS === "right" ? meanRightRef.current :
                        meanFwdRef.current;

            const qTwist = new THREE.Quaternion().setFromAxisAngle(
                twistAxis.clone().normalize(),
                THREE.MathUtils.degToRad(TWIST_DEG)
            );

// initial forward = meanFwd rotated by the chosen axis/angle
            const fwdInit = meanFwdRef.current.clone().applyQuaternion(qTwist);

// distance based on camera cluster size
            const camsPts = cams.map(c =>
                poseFromManifestC2W_OpenCV_to_Three(c.rotation, c.position, MANIFEST_IS_C2W).position
            );
            const gTmp = new THREE.BufferGeometry().setFromPoints(camsPts);
            gTmp.computeBoundingSphere();
            const r = gTmp.boundingSphere?.radius || 1;
            const dist = Math.max(1.25 * r, 0.6);

// lock yaw axis to mean-up
            yawAxisRef.current.copy(meanUp).normalize();

// place the initial view to face the subject (front view w/ zero roll)
            if (cameraRef.current && controlsRef.current) {
                const cam = cameraRef.current;
                const ctr = controlsRef.current;

                // forward with no roll: project meanFwd onto plane orthogonal to meanUp
                const forwardNoRoll = meanFwd.clone().sub(meanUp.clone().multiplyScalar(meanFwd.dot(meanUp))).normalize();
                const rGeom = new THREE.BufferGeometry().setFromPoints(
                    p.manifest.cameras.map(c => poseFromManifestC2W_OpenCV_to_Three(c.rotation, c.position, MANIFEST_IS_C2W).position)
                );
                rGeom.computeBoundingSphere();
                const R = rGeom.boundingSphere?.radius || 1;

                const dist = Math.max(1.25 * R, 0.6);
                const pos = center.clone().addScaledVector(forwardNoRoll, -dist);

                cam.up.copy(yawAxisRef.current);
                cam.position.copy(pos);
                cam.lookAt(center);
                ctr.target.copy(center);

                // we’re not using OrbitControls’ rotation anymore; keep zoom bounds sensible
                ctr.minDistance = Math.max(0.5 * R, 0.15);
                ctr.maxDistance = Math.max(4.0 * R, ctr.minDistance + 0.01);
                ctr.update();
                ctr.saveState();
            }

        }
    }, [p.manifest, camsOnDisk]); // runs on dataset/root cam availability changes

    /* ----- camera folders present on disk (reset on root change) ----- */
    useEffect(() => {
        let alive = true;
        setCurrentCamId(null);
        (async () => {
            try {
                const res = await fetch(`/api/cameras?session=${encodeURIComponent(p.session)}&root=${encodeURIComponent(rootDir)}`);
                const js = await res.json();
                const cams: string[] = js.cams ?? [];
                if (!alive) return;
                setCamsOnDisk(cams);
                setCurrentCamId(cams.length ? cams[0] : null);
            } catch {
                if (!alive) return;
                setCamsOnDisk([]);
                setCurrentCamId(null);
            }
        })();
        return () => {
            alive = false;
        };
    }, [p.session, rootDir]);

    /* ----- timestamps for current cam (keep playhead continuity) ----- */
    useEffect(() => {
        (async () => {
            if (!currentCamId) {
                setTsNames([]);
                return;
            }
            const url = `/api/timestamps?session=${encodeURIComponent(p.session)}&root=${encodeURIComponent(rootDir)}&cam=${encodeURIComponent(currentCamId)}`;
            try {
                const js = await (await fetch(url)).json();
                const all: string[] = (js.names || []).filter((s: string) => /^\d+$/.test(s));
                const startNum = p.start, endExclusive = p.start + p.total;
                const names = all.filter(s => {
                    const v = parseInt(s, 10);
                    return Number.isFinite(v) && v >= startNum && v < endExclusive;
                });
                const list = names.length ? names : all;
                setTsNames(list);

                // choose index closest to previous playhead timestamp (continuity)
                let idx = 0;
                const last = lastTsRef.current;
                if (last && list.length) {
                    const tgt = parseInt(last, 10);
                    let best = 0, bestDiff = Infinity;
                    for (let i = 0; i < list.length; i++) {
                        const v = parseInt(list[i], 10);
                        const d = Math.abs(v - tgt);
                        if (d < bestDiff) {
                            bestDiff = d;
                            best = i;
                        }
                    }
                    idx = best;
                }
                playIdxRef.current = idx;
                setStatus(`Camera ${currentCamId}: ${list.length} frames`);

                // show that frame immediately (request if needed)
                const name = list[idx];
                if (!pushFrameToGPUByName(name)) {
                    await requestFrame(name);
                    pushFrameToGPUByName(name);
                }
            } catch {
                setTsNames([]);
            }
        })();
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [p.session, rootDir, currentCamId, p.start, p.total]);

    /* ----- request + cache one frame via worker ----- */
    const requestFrame = useCallback(async (name: string): Promise<boolean> => {
        if (!workerRef.current || !currentCamId) return false;
        const url = `/reconstructions/${encodeURIComponent(p.session)}/${rootDir}/${currentCamId}/${name}.ply`;

        if (cacheRef.current.has(url)) return true;

        if (inflightRef.current.has(url)) {
            // wait until resolved
            return new Promise<boolean>((resolve) => {
                const poll = () => {
                    if (cacheRef.current.has(url)) resolve(true);
                    else if (!inflightRef.current.has(url)) resolve(false);
                    else setTimeout(poll, 20);
                };
                poll();
            });
        }

        if (inflightRef.current.size >= MAX_INFLIGHT) {
            await new Promise(r => setTimeout(r, 0));
            return requestFrame(name);
        }

        inflightRef.current.add(url);
        return new Promise<boolean>((resolve) => {
            resolversRef.current.set(url, resolve);
            workerRef.current!.postMessage({url, modality: p.modality, worldRot: null});
        });
    }, [p.session, rootDir, currentCamId, p.modality]);

    /* ----- upload to GPU (apply S flip to points here) ----- */
    const pushFrameToGPUByName = useCallback((name: string): boolean => {
        if (!currentCamId || !posAttrRef.current || !colAttrRef.current || !ptsRef.current) return false;
        const url = `/reconstructions/${encodeURIComponent(p.session)}/${rootDir}/${currentCamId}/${name}.ply`;
        const fr = cacheRef.current.get(url);
        if (!fr) return false;

        const posArr = posAttrRef.current.array as Float32Array;
        const colArr = colAttrRef.current.array as Uint8Array;
        const n = Math.min(fr.count, posArr.length / 3);

        const tmp = fr.positions.subarray(0, n * 3);
        const buf = new Float32Array(tmp);
        // OpenCV -> OpenGL
        for (let i = 0; i < buf.length; i += 3) {
            const x = buf[i], y = buf[i + 1], z = buf[i + 2];
            buf[i] = x * UNIT_SCALE;
            buf[i + 1] = -y * UNIT_SCALE;
            buf[i + 2] = -z * UNIT_SCALE;
        }

        posArr.set(buf, 0);
        colArr.set(fr.colors.subarray(0, n * 3), 0);
        (posAttrRef.current as any).updateRange = {offset: 0, count: n * 3};
        (colAttrRef.current as any).updateRange = {offset: 0, count: n * 3};
        posAttrRef.current.needsUpdate = true;
        colAttrRef.current.needsUpdate = true;
        ptsRef.current.geometry.setDrawRange(0, n);

        if (!centroidRef.current && n > 0 && posAttrRef.current) {
            const arr = posAttrRef.current.array as Float32Array;

            // Sample up to ~50k points for a stable centroid (fast)
            const stride = Math.max(1, Math.floor(n / 50000));
            let cx = 0, cy = 0, cz = 0, c = 0;
            for (let i = 0; i < n; i += stride) {
                const j = i * 3;
                cx += arr[j + 0];
                cy += arr[j + 1];
                cz += arr[j + 2];
                c++;
            }
            const centroid = new THREE.Vector3(cx / c, cy / c, cz / c);
            centroidRef.current = centroid;

            // Rough radius (max distance of a sparser subset)
            let r = 0;
            const strideR = stride * 4;
            for (let i = 0; i < n; i += strideR) {
                const j = i * 3;
                const dx = arr[j + 0] - centroid.x;
                const dy = arr[j + 1] - centroid.y;
                const dz = arr[j + 2] - centroid.z;
                const d = Math.sqrt(dx * dx + dy * dy + dz * dz);
                if (d > r) r = d;
            }
            radiusRef.current = Math.max(r, 0.4); // safety floor

        }

        lastTsRef.current = name; // continuity on cam switch
        return true;
    }, [p.session, rootDir, currentCamId]);

    /* ----- playback loop (fixed 30fps + small prefetch) ----- */
    useEffect(() => {
        if (!tsNames.length) return;

        // show current idx immediately
        (async () => {
            const name = tsNames[playIdxRef.current] ?? tsNames[0];
            if (!pushFrameToGPUByName(name)) {
                await requestFrame(name);
                pushFrameToGPUByName(name);
            }
        })();

        const tick = async () => {
            if (!isPlayingRef.current || !tsNames.length) {
                timerRef.current = window.setTimeout(tick, Math.round(1000 / TARGET_FPS));
                return;
            }
            const idx = playIdxRef.current;
            const name = tsNames[idx];

            if (!pushFrameToGPUByName(name)) {
                await requestFrame(name);
                pushFrameToGPUByName(name);
            }

            // prefetch ahead
            for (let a = 1; a <= PREFETCH_AHEAD; a++) {
                const ahead = tsNames[(idx + a) % tsNames.length];
                void requestFrame(ahead);
            }

            playIdxRef.current = (idx + 1) % tsNames.length;
            setStatus(`Playing ${playIdxRef.current + 1} / ${tsNames.length}`);
            timerRef.current = window.setTimeout(tick, Math.round(1000 / TARGET_FPS));
        };

        isPlayingRef.current = isPlaying;
        timerRef.current = window.setTimeout(tick, Math.round(1000 / TARGET_FPS));

        return () => {
            if (timerRef.current != null) {
                clearTimeout(timerRef.current);
                timerRef.current = null;
            }
        };
    }, [tsNames, isPlaying, requestFrame, pushFrameToGPUByName]);

    /* ----- auto-switch to nearest GT camera (with hysteresis) ----- */
    useEffect(() => {
        let stop = false;
        let current = currentCamId;

        const loop = () => {
            if (stop) return;
            if (!cameraRef.current || gtCamPoseRef.current.size === 0) {
                setTimeout(() => requestAnimationFrame(loop), Math.round(1000 / SWITCH_HZ));
                return;
            }

            const p = cameraRef.current.position;
            const curD = current && gtCamPoseRef.current.get(current)
                ? p.distanceToSquared(gtCamPoseRef.current.get(current)!.pos)
                : Infinity;

            let bestId: string | null = current ?? null;
            let bestD = curD;
            for (const [id, pose] of gtCamPoseRef.current.entries()) {
                const d = p.distanceToSquared(pose.pos);
                if (d < bestD) {
                    bestD = d;
                    bestId = id;
                }
            }

            if (bestId && bestId !== current && bestD < curD * (1 - SWITCH_HYSTERESIS)) {
                setCurrentCamId(bestId);
                current = bestId;
                // (We only swap which PLY is shown; we do not jump the viewer camera here.)
            }

            setTimeout(() => requestAnimationFrame(loop), Math.round(1000 / SWITCH_HZ));
        };
        loop();

        return () => {
            stop = true;
        };
    }, [currentCamId]);

    useEffect(() => {
        const prevBg = document.body.style.backgroundColor;
        document.body.style.backgroundColor = "#0B0B0B"; // dark background

        return () => {
            document.body.style.backgroundColor = prevBg; // restore previous
        };
    }, []);

    /* ----- UI ----- */
    return (
        <div ref={mountRef} className="w-full h-full bg-neutral-50">
            {/* overlay status */}
            <div
                className="absolute top-25 right-3 z-20 text-xs text-neutral-600 bg-white/85 backdrop-blur px-2 py-1 rounded shadow">
                {status || "Ready"}{currentCamId ? ` — ${currentCamId}` : ""}
            </div>

            {/* bottom transport */}
            <div
                className="absolute bottom-0 left-0 right-0 z-20 bg-white/90 border-t px-3 py-2 flex items-center gap-3">
                <button
                    onClick={() => {
                        setIsPlaying(v => !v);
                        isPlayingRef.current = !isPlayingRef.current;
                    }}
                    className="px-3 py-1 rounded border shadow-sm"
                    title={isPlaying ? "Pause" : "Play"}
                >
                    {isPlaying ? "⏸" : "▶️"}
                </button>

                <input
                    type="range"
                    min={0}
                    max={Math.max(0, tsNames.length - 1)}
                    value={playIdxRef.current}
                    onChange={async (e) => {
                        const v = parseInt(e.currentTarget.value, 10);
                        playIdxRef.current = v;
                        const name = tsNames[v];
                        if (!pushFrameToGPUByName(name)) await requestFrame(name);
                        setStatus(`Paused at ${v + 1} / ${tsNames.length}`);
                    }}
                    className="w-full"
                />

                <div className="text-xs text-neutral-700 select-none min-w-[140px] text-right">
                    {tsNames.length ? `${playIdxRef.current + 1} / ${tsNames.length}` : "0 / 0"}
                </div>
            </div>
        </div>
    );
}
