import fs from "node:fs/promises";
import path from "node:path";
import ReconGrid from "../components/ReconGrid";

export default async function Page() {
    const base = process.env.DATA_ROOT!;
    const sessions = (await fs.readdir(base, {withFileTypes: true}))
        .filter(e => e.isDirectory())
        .map(e => e.name)
        .filter(name => !name.includes("Calib"));

    const cards = await Promise.all(
        sessions.map(async (s) => {
            const reconDir = path.join(base, s, "reconstruction");
            try {
                const files = await fs.readdir(reconDir);
                const teaser = files.find(
                    f => f.startsWith("teaser_grid_") && f.endsWith(".mp4")
                );
                return {
                    session: s,
                    title: s.replaceAll("_", " "),
                    teaser: teaser ? `/reconstructions/${encodeURIComponent(s)}/${teaser}` : null,
                };
            } catch {
                return {session: s, title: s.replaceAll("_", " "), teaser: null};
            }
        })
    );

    return (
        <main className="p-6 max-w-6xl mx-auto pt-32">
            <h1 className="text-2xl font-semibold mb-6">Reconstructions</h1>
            <ReconGrid cards={cards}/>
        </main>
    );
}
