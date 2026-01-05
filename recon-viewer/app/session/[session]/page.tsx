import fs from "node:fs/promises";
import path from "node:path";
import SyncedTeaserGrid from "../../../components/SyncedTeaserGrid";

export const dynamic = "force-dynamic"; // ensure Node runtime; don't pre-render
export const revalidate = 0;

function pad4(n: number) {
    return String(n).padStart(4, "0");
}

function pad3(n: number) {
    return String(n).padStart(3, "0");
}

export default async function SessionPage({params}: { params: { session: string } }) {
    const session = decodeURIComponent(params.session);
    const base = process.env.DATA_ROOT!;
    const reconDir = path.join(base, session, "reconstruction");

    // Load manifest safely
    let manifest: any = {};
    try {
        const manifestPath = path.join(reconDir, "manifest.json");
        const txt = await fs.readFile(manifestPath, "utf8");
        manifest = JSON.parse(txt);
    } catch {
        manifest = {};
    }

    const ranges: any[] = Array.isArray(manifest.reconstructions) ? manifest.reconstructions : [];

    // Helper: pick a rootDir that actually exists for this range (best-effort)
    async function chooseRootDir(r: any): Promise<string | null> {
        const depthKey = r?.defaultDepthSource ?? Object.keys(r?.depthSources ?? {})[0];
        for (const rt of (r?.recon_types ?? [])) {
            const dsDir = r?.depthSources?.[depthKey]?.dir as string | undefined;
            const candidate = (dsDir && dsDir.startsWith(rt + "_")) ? dsDir : `${rt}_${depthKey}`;
            const teaserFile = path.join(
                reconDir,
                candidate,
                `teaser_${pad4(r.frameStart)}_${pad3(r.totalFrames)}.mp4`
            );
            try {
                const stat = await fs.stat(teaserFile);
                if (stat.isFile()) return candidate;
            } catch {
            }
        }
        return null;
    }

    // Build cards safely
    const cards = await Promise.all(
        ranges.map(async (r) => {
            try {
                const root = await chooseRootDir(r);
                const teaser = root
                    ? `/reconstructions/${encodeURIComponent(session)}/${root}/teaser_${pad4(r.frameStart)}_${pad3(r.totalFrames)}.mp4`
                    : null; // if we can't prove existence, leave null so client won't wait forever

                const dsKeys = Object.keys(r.depthSources ?? {});
                const initialRecon = (r.recon_types ?? [])[0] ?? "pcd";
                const initialDepth = r.defaultDepthSource ?? dsKeys[0] ?? "bilateral_temporal";

                return {
                    frameStart: r.frameStart,
                    totalFrames: r.totalFrames,
                    meta: `Types: ${(r.recon_types ?? []).join(", ")} · Depth: ${dsKeys.join(", ")}`,
                    teaser, // may be null
                    href: `/viewer/${encodeURIComponent(session)}/${r.frameStart}/${r.totalFrames}?recon=${initialRecon}&depth=${initialDepth}`,
                };
            } catch {
                return {
                    frameStart: r?.frameStart ?? 0,
                    totalFrames: r?.totalFrames ?? 0,
                    meta: "Unavailable",
                    teaser: null,
                    href: "#",
                };
            }
        })
    );

    const title = manifest?.title ?? session.replaceAll("_", " ");
    const subtitle =
        [manifest?.location, manifest?.date, manifest?.time].filter(Boolean).join(" — ") || null;

    return (
        <main className="p-6 max-w-5xl mx-auto">
            <h1 className="text-2xl font-semibold mb-2">{title}</h1>
            {subtitle && <p className="text-sm text-neutral-600 mb-6">{subtitle}</p>}

            {cards.length === 0 ? (
                <div className="rounded-xl border border-neutral-200 bg-white p-6 text-neutral-600">
                    No reconstructions found for this session.
                </div>
            ) : (
                <SyncedTeaserGrid cards={cards}/>
            )}
        </main>
    );
}
