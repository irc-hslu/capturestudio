import {NextResponse} from "next/server";
import fs from "node:fs/promises";
import path from "node:path";

export async function GET(req: Request) {
    const {searchParams} = new URL(req.url);
    const session = searchParams.get("session");
    const root = searchParams.get("root"); // e.g. "pcd_bilateral_temporal"
    if (!session || !root) return NextResponse.json({error: "Missing session or root"}, {status: 400});

    const base = process.env.DATA_ROOT;
    if (!base) return NextResponse.json({error: "DATA_ROOT not set"}, {status: 500});

    const dir = path.join(base, session, "reconstruction", root);
    try {
        const entries = await fs.readdir(dir, {withFileTypes: true});
        const cams = entries
            .filter(e => e.isDirectory() && /^cam/i.test(e.name))
            .map(e => e.name)
            .sort();
        return NextResponse.json({cams});
    } catch (e: any) {
        return NextResponse.json({error: e.message}, {status: 404});
    }
}
