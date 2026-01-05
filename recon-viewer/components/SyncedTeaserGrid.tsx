"use client";

import Link from "next/link";
import {useEffect, useMemo, useRef, useState} from "react";

type Card = {
    frameStart: number;
    totalFrames: number;
    meta: string;
    teaser: string | null;
    href: string;
};

function Spinner({className = ""}: { className?: string }) {
    return (
        <div
            aria-hidden
            className={`h-5 w-5 rounded-full border-2 border-neutral-300 border-t-transparent animate-spin ${className}`}
        />
    );
}

export default function SyncedTeaserGrid({cards}: { cards?: Card[] | null }) {
    // Always work with a real array to avoid undefined issues
    const items = useMemo<Card[]>(() => (Array.isArray(cards) ? cards : []), [cards]);

    const videoRefs = useRef<(HTMLVideoElement | null)[]>([]);
    useEffect(() => {
        // keep refs array length aligned to items length
        videoRefs.current = videoRefs.current.slice(0, items.length);
    }, [items.length]);

    // Ready/failed tracking
    const [readyMap, setReadyMap] = useState<Record<number, boolean>>({});
    const [failedMap, setFailedMap] = useState<Record<number, boolean>>({});

    const markReady = (i: number) => setReadyMap((p) => (p[i] ? p : {...p, [i]: true}));
    const markFailed = (i: number) =>
        setFailedMap((p) => (p[i] ? p : {...p, [i]: true}));

    const playableIdx = items
        .map((c, i) => ({i, playable: Boolean(c.teaser) && !failedMap[i]}))
        .filter((x) => x.playable)
        .map((x) => x.i);

    const totalVideos = items.filter((c) => Boolean(c.teaser)).length; // includes those that may fail
    const doneCount =
        Object.values(readyMap).filter(Boolean).length +
        Object.values(failedMap).filter(Boolean).length;
    const allReady = totalVideos === 0 || doneCount >= totalVideos;

    useEffect(() => {
        // Build the list of actual <video> elements we will sync (exclude failed)
        const vids = playableIdx
            .map((i) => videoRefs.current[i])
            .filter((v): v is HTMLVideoElement => !!v);

        if (vids.length === 0) return;

        // Mark cached-ready immediately
        vids.forEach((v) => {
            const idx = Number(v.dataset.idx ?? "-1");
            if (idx >= 0 && v.readyState >= 2) markReady(idx);
        });

        // Normalize config
        vids.forEach((v) => {
            v.muted = true;
            v.loop = true;
            v.playsInline = true;
            try {
                v.pause();
            } catch {
            }
            try {
                v.currentTime = 0;
            } catch {
            }
        });

        let raf: number | null = null;
        let started = false;

        const startAll = async () => {
            if (started) return;
            started = true;
            await Promise.all(vids.map((v) => v.play().catch(() => {
            })));

            const master = vids[0];
            const SYNC_EPS = 0.05; // seconds

            const tick = () => {
                const t = master.currentTime;
                for (const v of vids) {
                    if (v === master) continue;
                    const dur = isFinite(v.duration) && v.duration > 0 ? v.duration : null;
                    const target = dur ? (t % dur) : t;
                    const drift = Math.abs(v.currentTime - target);
                    if (drift > SYNC_EPS) {
                        try {
                            v.currentTime = target;
                        } catch {
                        }
                    }
                    if (v.paused) v.play().catch(() => {
                    });
                }
                raf = requestAnimationFrame(tick);
            };
            raf = requestAnimationFrame(tick);
        };

        if (allReady) startAll();

        const handleVis = () => {
            if (document.hidden) {
                vids.forEach((v) => v.pause());
            } else {
                const master = vids[0];
                const t = master.currentTime;
                vids.forEach((v) => {
                    const dur = isFinite(v.duration) && v.duration > 0 ? v.duration : null;
                    const target = dur ? (t % dur) : t;
                    try {
                        v.currentTime = target;
                    } catch {
                    }
                    v.play().catch(() => {
                    });
                });
            }
        };
        document.addEventListener("visibilitychange", handleVis);

        return () => {
            if (raf) cancelAnimationFrame(raf);
            document.removeEventListener("visibilitychange", handleVis);
        };
    }, [items, playableIdx.join(","), allReady]);

    return (
        <>
            {/* Global progress */}
            {!allReady && totalVideos > 0 && (
                <div
                    role="status"
                    aria-live="polite"
                    className="mb-4 flex items-center gap-3 rounded-xl border border-neutral-200 bg-white p-3 shadow-sm"
                >
                    <Spinner/>
                    <div className="flex-1">
                        <div className="text-sm font-medium">Preparing previews…</div>
                        <div className="mt-2 h-2 w-full overflow-hidden rounded bg-neutral-200">
                            <div
                                className="h-full bg-gray-900 transition-[width] duration-300"
                                style={{width: `${Math.round((doneCount / Math.max(1, totalVideos)) * 100)}%`}}
                            />
                        </div>
                    </div>
                    <div className="text-xs tabular-nums">
                        {doneCount}/{totalVideos}
                    </div>
                </div>
            )}

            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                {items.map((c, i) => {
                    const hasTeaser = Boolean(c.teaser);
                    const isReady = !!readyMap[i];
                    const isFailed = !!failedMap[i];

                    // @ts-ignore
                    return (
                        <Link
                            key={`${c.frameStart}-${c.totalFrames}-${i}`}
                            href={c.href}
                            className="group relative block overflow-hidden rounded-xl border border-neutral-200 bg-white shadow-sm transition-all hover:-translate-y-0.5 hover:shadow-lg hover:border-gray-600 focus-visible:-translate-y-0.5 focus-visible:shadow-lg focus-visible:border-gray-900"
                            tabIndex={0}
                        >
                            {/* Focus/hover ring */}
                            <span
                                aria-hidden
                                className="pointer-events-none absolute inset-0 rounded-xl ring-0 ring-gray-900/0 transition group-hover:ring-2 group-hover:ring-gray-800 group-focus-visible:ring-2 group-focus-visible:ring-gray-900"
                            />

                            {/* Video area */}
                            <div className="relative w-full aspect-[4/3] bg-black">
                                {hasTeaser && !isFailed ? (
                                    <video
                                        ref={(el: HTMLVideoElement | null) => {
                                            videoRefs.current[i] = el
                                        }}
                                        data-idx={i}
                                        src={c.teaser!}
                                        preload="metadata"
                                        muted
                                        playsInline
                                        className="h-auto w-full object-contain pointer-events-none"
                                        onCanPlay={() => markReady(i)}
                                        onError={() => markFailed(i)}
                                    />
                                ) : (
                                    <div
                                        className="h-full w-full bg-neutral-200 grid place-items-center text-xs text-neutral-500">
                                        {isFailed ? "Preview unavailable" : "No preview"}
                                    </div>
                                )}

                                {/* Per-card loading overlay */}
                                {hasTeaser && !isFailed && !isReady && (
                                    <div className="absolute inset-0 grid place-items-center bg-neutral-900/30">
                                        <Spinner className="h-6 w-6"/>
                                        <span className="sr-only">Loading video preview…</span>
                                    </div>
                                )}
                            </div>

                            {/* Footer */}
                            <div className="p-3 flex items-center justify-between">
                                <div>
                                    <div className="font-medium">
                                        Frames {c.frameStart} → {c.frameStart + c.totalFrames}
                                    </div>
                                    <div className="text-xs text-neutral-500">{c.meta}</div>
                                </div>
                                <Link
                                    href={c.href}
                                    className="px-3 py-1.5 rounded bg-black text-white text-sm transition
                             hover:bg-black focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-gray-900"
                                >
                                    View
                                </Link>
                            </div>

                            {/* Subtle overlay on hover/focus */}
                            <div
                                aria-hidden
                                className="pointer-events-none absolute inset-0 opacity-0 group-hover:opacity-100 group-focus-visible:opacity-100 bg-gradient-to-t from-black/10 to-transparent transition"
                            />
                        </Link>
                    );
                })}
            </div>
        </>
    );
}
