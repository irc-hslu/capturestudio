import fs from "node:fs/promises";
import path from "node:path";
import crypto from "node:crypto";
import {NextResponse} from "next/server";

const MIME: Record<string, string> = {
    ".ply": "model/x.ply",
    ".mp4": "video/mp4",
    ".json": "application/json",
};

export async function GET(req: Request, ctx: { params: { path: string[] } }) {
    const {pathname} = new URL(req.url);
    // URL form: /reconstructions/<SESSION>/<...rest>
    const [, , session, ...restParts] = pathname.split("/");
    const rest = restParts.join("/");

    const baseRoot = process.env.DATA_ROOT!;
    const baseAbs = path.resolve(baseRoot, decodeURIComponent(session), "reconstruction"); // SINGULAR
    const fileAbs = path.resolve(baseAbs, decodeURIComponent(rest));

    if (!fileAbs.startsWith(baseAbs)) {
        return NextResponse.json({error: "Forbidden"}, {status: 403});
    }

    try {
        const data = await fs.readFile(fileAbs);
        const stat = await fs.stat(fileAbs);

        const etag = crypto
            .createHash("sha1")
            .update(String(stat.size) + "-" + String(stat.mtimeMs))
            .digest("hex");

        const ifNoneMatch = req.headers.get("if-none-match");
        if (ifNoneMatch && ifNoneMatch === etag) {
            return new NextResponse(null, {
                status: 304,
                headers: {
                    "Cache-Control": "public, max-age=31536000, immutable",
                    ETag: etag,
                },
            });
        }

        const ext = path.extname(fileAbs).toLowerCase();
        const ct = MIME[ext] || "application/octet-stream";

        // @ts-ignore
        return new NextResponse(data, {
            status: 200,
            headers: {
                "Content-Type": ct,
                "Cache-Control": "public, max-age=31536000, immutable",
                ETag: etag,
                "Last-Modified": new Date(stat.mtimeMs).toUTCString(),
                "Accept-Ranges": "bytes",
            },
        });
    } catch (e: any) {
        return NextResponse.json({error: e.message}, {status: 404});
    }
}
