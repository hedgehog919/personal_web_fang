// ResearchTab.tsx 研究經驗頁面
// 設定研究經驗的資料來源，含 id、title、author、department、advisor、date、sections 等欄位
'use client';

import { useState } from 'react';
import { research } from '@/data/research';
import ReactMarkdown from 'react-markdown';

export default function ResearchTab() {
    const [activeSection, setActiveSection] = useState<string>('abstract');

    const scrollToSection = (sectionId: string) => {
        setActiveSection(sectionId);
        const element = document.getElementById(sectionId);
        if (element) {
            element.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }
    };

    return (
        <div className="flex gap-6">
            {/* Left Sidebar - Table of Contents */}
            <aside className="hidden lg:block w-48 flex-shrink-0">
                <div className="sticky top-4">
                    <h3 className="text-sm font-semibold text-slate-400 uppercase tracking-wider mb-4">
                        目錄
                    </h3>
                    <nav className="space-y-1">
                        {research.sections.map((section) => (
                            <button
                                key={section.id}
                                onClick={() => scrollToSection(section.id)}
                                className={`block w-full text-left px-3 py-2 rounded-lg text-sm transition-colors ${activeSection === section.id
                                    ? 'bg-blue-50 text-blue-700 font-medium'
                                    : 'text-slate-600 hover:text-blue-600 hover:bg-slate-50'
                                    }`}
                            >
                                {section.title}
                            </button>
                        ))}
                    </nav>
                </div>
            </aside>

            {/* Main Content - Paper Style */}
            <main className="flex-1 min-w-0">
                {/* Paper Header */}
                <header className="text-center mb-12 pb-8 border-b border-slate-200">
                    <h1 className="text-2xl md:text-3xl font-bold text-slate-900 mb-4 leading-tight">
                        {research.title}
                    </h1>
                    <div className="space-y-2 text-slate-600">
                        <p className="text-lg">{research.author}</p>
                        <p className="text-sm text-slate-500">{research.department}</p>
                        <p className="text-sm text-slate-500">指導教授：{research.advisor}</p>
                        <p className="text-sm text-slate-500">{research.date}</p>
                    </div>
                </header>

                {/* Mobile Table of Contents */}
                <div className="lg:hidden mb-8">
                    <details className="bg-slate-50 rounded-lg border border-slate-200">
                        <summary className="px-4 py-3 cursor-pointer text-slate-600 font-medium">
                            📑 目錄
                        </summary>
                        <nav className="px-4 pb-4 space-y-1">
                            {research.sections.map((section) => (
                                <button
                                    key={section.id}
                                    onClick={() => scrollToSection(section.id)}
                                    className="block w-full text-left px-3 py-2 rounded-lg text-sm text-slate-600 hover:text-blue-600 hover:bg-slate-50 transition-colors"
                                >
                                    {section.title}
                                </button>
                            ))}
                        </nav>
                    </details>
                </div>

                {/* Paper Sections */}
                <article className="space-y-12">
                    {research.sections.map((section, index) => (
                        <section
                            key={section.id}
                            id={section.id}
                            className="scroll-mt-4"
                        >
                            <h2 className="text-xl md:text-2xl font-bold text-blue-700 mb-6 pb-2 border-b border-slate-200">
                                {index + 1}. {section.title}
                            </h2>
                            <div className="prose prose-slate max-w-none">
                                <div className="text-slate-700 leading-relaxed space-y-4 research-content">
                                    <ReactMarkdown
                                        components={{
                                            h3: ({ children }) => (
                                                <h3 className="text-lg font-semibold text-slate-900 mt-6 mb-3">
                                                    {children}
                                                </h3>
                                            ),
                                            h4: ({ children }) => (
                                                <h4 className="text-base font-semibold text-slate-800 mt-4 mb-2">
                                                    {children}
                                                </h4>
                                            ),
                                            p: ({ children }) => (
                                                <p className="text-slate-700 leading-relaxed mb-4">
                                                    {children}
                                                </p>
                                            ),
                                            ul: ({ children }) => (
                                                <ul className="list-disc list-inside space-y-2 text-slate-700 ml-4">
                                                    {children}
                                                </ul>
                                            ),
                                            ol: ({ children }) => (
                                                <ol className="list-decimal list-inside space-y-2 text-slate-700 ml-4">
                                                    {children}
                                                </ol>
                                            ),
                                            li: ({ children }) => (
                                                <li className="text-slate-700">{children}</li>
                                            ),
                                            strong: ({ children }) => (
                                                <strong className="text-slate-900 font-semibold">{children}</strong>
                                            ),
                                            em: ({ children }) => (
                                                <em className="text-slate-800 italic">{children}</em>
                                            ),
                                            code: ({ children }) => (
                                                <code className="px-1.5 py-0.5 bg-slate-100 rounded text-blue-700 text-sm border border-slate-200">
                                                    {children}
                                                </code>
                                            ),
                                            blockquote: ({ children }) => (
                                                <blockquote className="border-l-4 border-blue-500 pl-4 italic text-slate-500">
                                                    {children}
                                                </blockquote>
                                            ),
                                        }}
                                    >
                                        {section.content}
                                    </ReactMarkdown>
                                </div>
                            </div>
                        </section>
                    ))}
                </article>

                {/* Back to Top */}
                <div className="mt-12 pt-8 border-t border-slate-700 text-center">
                    <button
                        onClick={() => scrollToSection('abstract')}
                        className="px-4 py-2 bg-slate-700 hover:bg-slate-600 text-slate-300 rounded-lg text-sm transition-colors"
                    >
                        ↑ 回到頂部
                    </button>
                </div>
            </main>
        </div>
    );
}