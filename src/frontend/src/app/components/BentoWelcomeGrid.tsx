"use client";
import { BookOpen, Sparkles, Star, Heart, ArrowUpRight } from "lucide-react";
import type { LucideIcon } from "lucide-react";

interface BentoCard {
    icon: LucideIcon;
    question: string;
    description: string;
    colSpan: string;
    iconColor: string;
    iconBg: string;
    delay: string;
}

const CARDS: BentoCard[] = [
    {
        icon: BookOpen,
        question: "What is the significance of the Urim and Thummim?",
        description: "Prophetic instruments and the Restoration",
        colSpan: "col-span-2",
        iconColor: "text-[#7c6cfc]",
        iconBg: "bg-[#7c6cfc]/20",
        delay: "0ms",
    },
    {
        icon: Sparkles,
        question: "Explain the Word of Wisdom and its health principles",
        description: "Doctrine & Covenants 89",
        colSpan: "col-span-1",
        iconColor: "text-emerald-500",
        iconBg: "bg-emerald-500/20",
        delay: "80ms",
    },
    {
        icon: Star,
        question: "What does the Topical Guide say about faith?",
        description: "Faith, hope, and works",
        colSpan: "col-span-1",
        iconColor: "text-amber-500",
        iconBg: "bg-amber-500/20",
        delay: "160ms",
    },
    {
        icon: Heart,
        question: "How does the Law of Chastity relate to temple covenants?",
        description: "Purity, covenant living, and eternal progression",
        colSpan: "col-span-2",
        iconColor: "text-rose-500",
        iconBg: "bg-rose-500/20",
        delay: "240ms",
    },
];

interface Props {
    onSelect: (question: string) => void;
    disabled: boolean;
}

export function BentoWelcomeGrid({ onSelect, disabled }: Props) {
    return (
        <div className="flex min-h-full flex-col items-center justify-center px-6 py-12">
            <div className="mb-10 text-center">
                <div className="mb-4 inline-flex items-center gap-2 rounded-full border border-[#7c6cfc]/20 bg-[#7c6cfc]/10 px-3.5 py-1.5 backdrop-blur-sm">
                    <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-[#7c6cfc]" />
                    <span className="text-xs font-medium text-[#7c6cfc]">LDS Doctrinal RAG</span>
                </div>
                <h1 className="text-2xl font-semibold tracking-tight text-slate-800 dark:text-neutral-100 sm:text-3xl">
                    What would you like to study?
                </h1>
            </div>

            <div className="grid w-full max-w-3xl grid-cols-3 gap-3">
                {CARDS.map(({ icon: Icon, question, description, colSpan, iconColor, iconBg, delay }) => (
                    <button
                        key={question}
                        type="button"
                        onClick={() => onSelect(question)}
                        disabled={disabled}
                        style={{ animationDelay: delay }}
                        className={`group animate-fade-up relative ${colSpan} flex flex-col gap-4 overflow-hidden rounded-2xl p-5 text-left disabled:cursor-not-allowed disabled:opacity-50
              bg-white/50 backdrop-blur-xl border border-white/60 shadow-sm hover:shadow-md
              dark:bg-white/[0.03] dark:backdrop-blur-xl dark:border-violet-500/20 hover:dark:border-violet-500/50
              transition-all`}
                    >
                        <div className="flex items-start justify-between relative z-10">
                            <div className={`flex h-9 w-9 items-center justify-center rounded-xl backdrop-blur-md ${iconBg}`}>
                                <Icon size={18} className={iconColor} />
                            </div>
                            <ArrowUpRight size={14} className="text-slate-400 group-hover:text-slate-600 dark:text-neutral-500 dark:group-hover:text-neutral-300 transition-colors" />
                        </div>

                        <div className="relative z-10 text-left">
                            <p className="text-sm font-medium leading-snug text-slate-800 dark:text-neutral-100">{question}</p>
                            <p className="mt-1 text-xs text-slate-600 dark:text-neutral-400">{description}</p>
                        </div>
                    </button>
                ))}
            </div>
        </div>
    );
}
