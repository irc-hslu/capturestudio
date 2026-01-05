import {NextResponse} from "next/server";
import fs from "node:fs/promises";
import path from "node:path";

export async function GET(req: Request) {
    const {searchParams} = new URL(req.url);
    const session = searchParams.get("session");
    const root = searchParams.get("root");
    const cam = searchParams.get("cam");
    if (!session || !root || !cam) return NextResponse.json({error: "Missing"}, {status: 400});

    const dir = path.join(process.env.DATA_ROOT!, session, "reconstruction", root, cam);
    try {
        const files = await fs.readdir(dir);
        const names = files
            .filter(f => f.endsWith(".ply"))
            .map(f => f.slice(0, -4))                   // "000123"
            .sort((a, b) => parseInt(a, 10) - parseInt(b, 10));
        return NextResponse.json({names, pad: names[0]?.length ?? 0});
    } catch (e: any) {
        return NextResponse.json({error: e.message}, {status: 404});
    }
}
