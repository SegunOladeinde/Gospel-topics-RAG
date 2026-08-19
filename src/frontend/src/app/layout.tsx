import type { Metadata, Viewport } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import { Providers } from "./providers";
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
  title: "Gospel Topics RAG",
  description:
    "Doctrinal Q&A grounded in LDS source texts, powered by AI retrieval-augmented generation.",
};

// Ensures the mobile browser doesn't zoom, and that the layout fills the full
// dynamic viewport (including behind Safari's URL bar on iOS).
export const viewport: Viewport = {
  width: "device-width",
  initialScale: 1,
  maximumScale: 1,
};

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html
      lang="en"
      className={`${geistSans.variable} ${geistMono.variable} min-h-full antialiased`}
      suppressHydrationWarning
    >
      <body
        className="bg-slate-50 bg-[radial-gradient(ellipse_at_top_right,_var(--tw-gradient-stops))] from-violet-100 via-white to-blue-50 dark:bg-[#0a0a0f] dark:bg-[radial-gradient(ellipse_at_top,_var(--tw-gradient-stops))] dark:from-violet-900/20 dark:via-[#0a0a0f] dark:to-[#0a0a0f] text-slate-800 dark:text-neutral-100 antialiased min-h-full overflow-hidden"
        suppressHydrationWarning
      >
        <Providers>{children}</Providers>
      </body>
    </html>
  );
}
