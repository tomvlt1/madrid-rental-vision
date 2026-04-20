import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import Link from "next/link";
import "./globals.css";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "CasaIntel — Agency dashboard",
  description: "Portfolio intelligence for Madrid rental agencies.",
};

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en" className={`${geistSans.variable} ${geistMono.variable} h-full antialiased`}>
      <body className="min-h-full flex flex-col">
        <header className="border-b border-slate-200 bg-white/90 backdrop-blur sticky top-0 z-10">
          <div className="mx-auto max-w-7xl px-6 py-4 flex items-center justify-between">
            <Link href="/" className="font-semibold tracking-tight text-slate-900">
              <span className="text-teal-700">Casa</span>Intel
              <span className="ml-2 text-xs font-normal text-slate-500">Madrid</span>
            </Link>
            <nav className="flex gap-1 text-sm">
              <NavLink href="/">Portfolio</NavLink>
              <NavLink href="/intake">Intake</NavLink>
            </nav>
          </div>
        </header>
        <main className="flex-1 mx-auto w-full max-w-7xl px-6 py-8">{children}</main>
        <footer className="border-t border-slate-200 bg-white py-4">
          <div className="mx-auto max-w-7xl px-6 text-xs text-slate-500">
            Trained on ~1,500 Madrid rental listings · ResNet-50 + sentence-transformers + gradient boosting · demo build.
          </div>
        </footer>
      </body>
    </html>
  );
}

function NavLink({ href, children }: { href: string; children: React.ReactNode }) {
  return (
    <Link
      href={href}
      className="px-3 py-1.5 rounded-md text-slate-700 hover:bg-slate-100 hover:text-slate-900 transition-colors"
    >
      {children}
    </Link>
  );
}
