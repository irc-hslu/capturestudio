// app/_components/ReconGrid.tsx
"use client";

import Link from "next/link";
import {useEffect, useMemo, useRef, useState} from "react";

type Card = { session: string; title: string; teaser: string | null };

function Spinner({className = ""}: { className?: string }) {
    return (
        <div
            aria-hidden
            className={`h-5 w-5 rounded-full border-2 border-neutral-300 border-t-transparent animate-spin ${className}`}
        />
    );
}

export default function ReconGrid({cards}: { cards: Card[] }) {
    const videoRefs = useRef<(HTMLVideoElement | null)[]>([]);
    const hasVideos = useMemo(() => cards.map(c => Boolean(c.teaser)), [cards]);

    // Track which indices have reported "ready to play".
    const [readyMap, setReadyMap] = useState<Record<number, boolean>>({});
    const totalVideos = hasVideos.filter(Boolean).length;
    const totalReady = Object.values(readyMap).filter(Boolean).length;
    const allReady = totalVideos === 0 || totalReady >= totalVideos;

    // Mark one index as ready (idempotent).
    const markReady = (i: number) =>
        setReadyMap(prev => (prev[i] ? prev : {...prev, [i]: true}));

    useEffect(() => {
        const vids = videoRefs.current.filter(
            (v): v is HTMLVideoElement => !!v
        );

        if (vids.length === 0) return;

        // If any video is already ready from cache, mark it immediately.
        vids.forEach(v => {
            const idx = Number(v.dataset.idx ?? "-1");
            if (idx >= 0 && v.readyState >= 2) markReady(idx);
        });

        // Ensure all videos are set up consistently.
        vids.forEach(v => {
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

            await Promise.all(vids.map(v => v.play().catch(() => {
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

        // Start together once ALL are ready.
        if (allReady) startAll();

        // Pause on tab hide; resync & resume on visible.
        const handleVis = () => {
            if (document.hidden) {
                vids.forEach(v => v.pause());
            } else {
                const master = vids[0];
                const t = master.currentTime;
                vids.forEach(v => {
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
    }, [hasVideos.join("|"), allReady]);

    return (
        <>
            {/* Global loading/progress while waiting for ALL teasers */}
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
                                style={{width: `${Math.round((totalReady / totalVideos) * 100)}%`}}
                            />
                        </div>
                    </div>
                    <div className="text-xs tabular-nums">
                        {totalReady}/{totalVideos}
                    </div>
                </div>
            )}

            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-6">
                {cards.map((c, i) => {
                    const isVideo = Boolean(c.teaser);
                    const isReady = readyMap[i];

                    return (
                        <Link
                            key={c.session}
                            href={`/session/${encodeURIComponent(c.session)}`}
                            className="group relative block overflow-hidden rounded-xl border border-neutral-200 bg-white shadow-sm transition-all hover:-translate-y-0.5 hover:shadow-lg hover:border-gray-600 focus-visible:-translate-y-0.5 focus-visible:shadow-lg focus-visible:border-gray-900 outline-none"
                        >
                            {/* Focus/hover ring */}
                            <span
                                aria-hidden
                                className="pointer-events-none absolute inset-0 rounded-xl ring-0 ring-gray-900/0 transition group-hover:ring-2 group-hover:ring-gray-700 group-focus-visible:ring-2 group-focus-visible:ring-gray-900"
                            />

                            {/* Video area */}
                            <div className="relative w-full aspect-[4/3] bg-black">
                                {c.teaser ? (
                                    <video
                                        ref={(el: HTMLVideoElement | null) => {
                                            videoRefs.current[i] = el
                                        }}
                                        data-idx={i}
                                        src={c.teaser}
                                        preload="metadata"
                                        muted
                                        playsInline
                                        // we start all together once all are ready
                                        className="h-auto w-full object-contain"
                                        onCanPlay={() => markReady(i)}
                                    />
                                ) : (
                                    <div className="h-auto w-full bg-neutral-200"/>
                                )}

                                {/* Per-card loading overlay */}
                                {isVideo && !isReady && (
                                    <div className="absolute inset-0 grid place-items-center bg-neutral-900/30">
                                        <Spinner className="h-6 w-6"/>
                                        <span className="sr-only">Loading video preview…</span>
                                    </div>
                                )}
                            </div>

                            {/* Title bar */}
                            <div className="p-3 font-medium">{c.title}</div>

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
