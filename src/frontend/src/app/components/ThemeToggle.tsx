"use client";
import { useTheme } from "next-themes";
import { Sun, Moon } from "lucide-react";
import { useEffect, useState } from "react";

export function ThemeToggle({ iconOnly = false }: { iconOnly?: boolean }) {
    const { theme, setTheme } = useTheme();
    const [mounted, setMounted] = useState(false);

    useEffect(() => setMounted(true), []);
    if (!mounted) return null;

    const isDark = theme === "dark";

    return (
        <button
            onClick={() => setTheme(isDark ? "light" : "dark")}
            aria-label={isDark ? "Switch to light mode" : "Switch to dark mode"}
            className="flex items-center gap-2 rounded-lg p-2 text-slate-500 transition-all duration-200 
        hover:bg-black/5 hover:text-slate-800 
        dark:text-neutral-400 dark:hover:bg-white/[0.08] dark:hover:text-neutral-200"
        >
            {isDark ? (
                <Sun size={15} className="text-amber-300 shrink-0" />
            ) : (
                <Moon size={15} className="text-[#7c6cfc] shrink-0" />
            )}
            {!iconOnly && (
                <span className="text-xs font-medium">
                    {isDark ? "Light Mode" : "Dark Mode"}
                </span>
            )}
        </button>
    );
}
