import "./globals.css";
import Link from "next/link";
import Image from "next/image";

export default function RootLayout({children}: { children: React.ReactNode }) {
    return (
        <html lang="en">
        <body className="min-h-screen bg-neutral-50">
        {/* Top bar with logo on the right */}
        <header className="fixed top-0 left-0 right-0 z-40 border-b bg-white/80 backdrop-blur">
            <div className="mx-auto max-w-6xl p-4">
                <div className="flex items-center justify-start">
                    <Link
                        href="/"
                        className="inline-flex items-center rounded p-2 transition hover:bg-neutral-100 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-gray-900"
                        aria-label="Home"
                    >
                        <Image
                            src="/logo.svg"   // change if you used .png
                            alt="HSLU"
                            width={200}
                            height={80}
                            priority
                        />
                    </Link>
                </div>
            </div>
        </header>

        {children}
        </body>
        </html>
    );
}
