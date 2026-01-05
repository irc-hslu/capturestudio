import fs from "node:fs/promises";
import path from "node:path";
import ThreeViewer from "@/components/ThreeViewer";

export default async function ViewerPage({params, searchParams}: {
    params: { session: string; start: string; total: string };
    searchParams: { recon?: "pcd" | "gs"; depth?: string; modality?: string };
}) {
    const {session, start, total} = params;
    const recon = searchParams.recon ?? "pcd";
    const depth = searchParams.depth ?? "bilateral_temporal";
    const modality = searchParams.modality ?? "color";

    const manifestPath = path.join(process.env.DATA_ROOT || '', session, "reconstruction", "manifest.json");
    const manifest = JSON.parse(await fs.readFile(manifestPath, "utf8"));

    const urlBase = `/viewer/${session}/${start}/${total}`;
    const mkHref = (q: Record<string, string>) =>
        urlBase + "?" + new URLSearchParams({recon, depth, modality, ...q}).toString();

    // @ts-ignore
    return (
        <main className="w-full h-[calc(100dvh-5px)] relative pt-26">
            {/* Info */}
            <div className="absolute top-25 left-3 z-10 bg-white/80 backdrop-blur rounded px-3 py-2 text-sm">
                <div className="font-medium">{manifest.title ?? session}</div>
                <div>{(manifest.performers || []).join(", ")}</div>
                <div className="text-neutral-600">{manifest.location} — {manifest.date} {manifest.time}</div>
                <div className="text-neutral-600">FPS {manifest.fps}</div>
            </div>
            <div className="absolute top-50 left-3 z-10 bg-white/80 backdrop-blur rounded px-3 py-2 text-sm space-x-2">
                {/* You’ll wire these to URL params on change */}
                <span>Depth:</span><strong>{depth}</strong>
                <span className="ml-3">Modality:</span><strong>{modality}</strong>
            </div>

            {/* Three.js viewer */}
            <ThreeViewer
                session={session}
                start={parseInt(start, 10)}
                total={parseInt(total, 10)}
                recon={recon}
                depth={depth}
                modality={modality}
                manifest={manifest}
            />
        </main>
    );
}
