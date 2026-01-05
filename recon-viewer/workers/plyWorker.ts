/// <reference lib="webworker" />
import {PLYLoader} from "three/examples/jsm/loaders/PLYLoader.js";

/* ---------------- Types ---------------- */
type WorkerRequest = {
    url: string;
    modality?: "color" | "depth" | "normals" | string;
    /** Optional point downsampling stride (1 = keep all) */
    stride?: number;
};

type WorkerResponseOk = {
    ok: true;
    url: string;
    positions: ArrayBufferLike;
    colors: ArrayBufferLike;
    normals?: ArrayBufferLike;
    count: number;
};

type WorkerResponseErr = {
    ok: false;
    url: string;
    error: string;
};

/* ---------------- Loader ---------------- */
const loader = new PLYLoader();

/* ---------------- Helpers ---------------- */

function makeUint8ColorsFromAttribute(attr: any, count: number, stride: number): Uint8Array {
    // Handles both Uint8 and Float attributes (normalized or not)
    const out = new Uint8Array(Math.ceil(count / stride) * 3);

    // Fast path when underlying array is Uint8Array of length 3*N
    const arr = attr?.array as ArrayLike<number> | undefined;
    const isU8 = arr instanceof Uint8Array && attr.itemSize === 3;
    if (isU8 && stride === 1) {
        out.set(arr as Uint8Array);
        return out;
    }

    // Generic path via getX/Y/Z (safe for Float32 attributes)
    let j = 0;
    for (let i = 0; i < count; i += stride) {
        const r = attr ? attr.getX(i) : 0.78;
        const g = attr ? attr.getY(i) : 0.78;
        const b = attr ? attr.getZ(i) : 0.78;

        // If attr is float-normalized 0..1, this works; if already 0..255, clamp handles it
        out[j++] = Math.max(0, Math.min(255, Math.round(r * (r <= 1 ? 255 : 1))));
        out[j++] = Math.max(0, Math.min(255, Math.round(g * (g <= 1 ? 255 : 1))));
        out[j++] = Math.max(0, Math.min(255, Math.round(b * (b <= 1 ? 255 : 1))));
    }
    return out;
}

function makeNormalsArray(attr: any, count: number, stride: number): Float32Array | undefined {
    if (!attr) return undefined;
    const out = new Float32Array(Math.ceil(count / stride) * 3);
    let j = 0;
    for (let i = 0; i < count; i += stride) {
        out[j++] = attr.getX(i);
        out[j++] = attr.getY(i);
        out[j++] = attr.getZ(i);
    }
    return out;
}

function colorsFromDepth(positions: Float32Array): Uint8Array {
    // Simple grayscale from Z (min-max normalized)
    const n = positions.length / 3;
    let zmin = Infinity, zmax = -Infinity;
    for (let i = 2; i < positions.length; i += 3) {
        const z = positions[i];
        if (z < zmin) zmin = z;
        if (z > zmax) zmax = z;
    }
    const range = zmax - zmin || 1;
    const out = new Uint8Array(n * 3);
    for (let i = 0, j = 2, k = 0; i < n; i++, j += 3, k += 3) {
        const z = positions[j];
        const t = (z - zmin) / range;
        const g = Math.max(0, Math.min(255, Math.round(t * 255)));
        out[k] = g;
        out[k + 1] = g;
        out[k + 2] = g;
    }
    return out;
}

function colorsFromNormals(normals: Float32Array): Uint8Array {
    const n = normals.length / 3;
    const out = new Uint8Array(n * 3);
    for (let i = 0, k = 0; i < n; i++, k += 3) {
        const nx = normals[k], ny = normals[k + 1], nz = normals[k + 2];
        out[k] = Math.round((nx * 0.5 + 0.5) * 255);
        out[k + 1] = Math.round((ny * 0.5 + 0.5) * 255);
        out[k + 2] = Math.round((nz * 0.5 + 0.5) * 255);
    }
    return out;
}

/* ---------------- Worker ---------------- */

self.onmessage = async (ev: MessageEvent<WorkerRequest>) => {
    const {url, modality = "color", stride = 1} = ev.data;

    try {
        // Let the browser reuse resources when headers allow it (prod)
        const res = await fetch(url, {
            cache: "force-cache",
            credentials: "same-origin",
            mode: "same-origin",
        });
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const buf = await res.arrayBuffer();

        // Parse PLY (handles ascii/binary)
        const geom = loader.parse(buf);
        const posAttr: any = geom.getAttribute("position");
        if (!posAttr) throw new Error("PLY has no 'position' attribute");

        const colorAttr: any = geom.getAttribute("color");
        const normalAttr: any = geom.getAttribute("normal");

        const N = posAttr.count;
        const s = Math.max(1, Math.floor(stride));
        const outCount = Math.ceil(N / s);

        // Positions (no alignment here; send raw OpenCV coords to main thread)
        const positions = new Float32Array(outCount * 3);
        for (let i = 0, j = 0; i < N; i += s, j += 3) {
            positions[j] = posAttr.getX(i);
            positions[j + 1] = posAttr.getY(i);
            positions[j + 2] = posAttr.getZ(i);
        }

        // Normals (optional)
        const normals = normalAttr ? makeNormalsArray(normalAttr, N, s) : undefined;

        // Colors
        let colors: Uint8Array;
        if (modality === "color") {
            colors = makeUint8ColorsFromAttribute(colorAttr, N, s);
        } else if (modality === "depth") {
            colors = colorsFromDepth(positions);
        } else if (modality === "normals" && normals) {
            colors = colorsFromNormals(normals);
        } else {
            // Fallback neutral gray
            colors = new Uint8Array(outCount * 3);
            colors.fill(200);
        }

        // Return with transferables and echo URL
        const okMsg: WorkerResponseOk = {
            ok: true,
            url,
            positions: positions.buffer,
            colors: colors.buffer,
            normals: normals?.buffer,
            count: outCount,
        };

        const transfers: ArrayBufferLike[] = [positions.buffer, colors.buffer];
        if (normals) transfers.push(normals.buffer);
        (self as unknown as Worker).postMessage(okMsg, transfers);
    } catch (e: any) {
        const errMsg: WorkerResponseErr = {ok: false, url, error: String(e?.message || e)};
        (self as unknown as Worker).postMessage(errMsg);
    }
};
