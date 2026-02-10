// WorksTab.tsx 工作經驗頁面
// 負責顯示工作經驗的卡片、排序、篩選等功能
'use client'; // 瀏覽器端互動元件

// 
import { useState } from 'react';
import { Github, Calendar, Building2, ChevronDown, Briefcase, GraduationCap } from 'lucide-react';
import { works } from '@/data/works'; // 引入工作經驗的資料

type SortOrder = 'newest' | 'oldest';
type Category = 'all' | 'PlanningAssistant' | 'FullTime';

export default function WorksTab() {
    const [expandedWork, setExpandedWork] = useState<string | null>(null);
    const [sortOrder, setSortOrder] = useState<SortOrder>('newest');
    const [category, setCategory] = useState<Category>('all');

    const toggleExpand = (workId: string) => {
        setExpandedWork(expandedWork === workId ? null : workId);
    };

    // 解析日期並排序
    const parseDate = (period: string): Date => {
        const endDate = period.split(' - ')[1] || period.split(' - ')[0];
        const [month, year] = endDate.split(' ');
        const monthMap: Record<string, number> = {
            'Jan': 0, 'Feb': 1, 'Mar': 2, 'Apr': 3, 'May': 4, 'Jun': 5,
            'Jul': 6, 'Aug': 7, 'Sep': 8, 'Oct': 9, 'Nov': 10, 'Dec': 11
        };
        return new Date(parseInt(year), monthMap[month] || 0);
    };

    // 篩選並排序
    const filteredWorks = works
        .filter(work => category === 'all' || work.category === category)
        .sort((a, b) => {
            const dateA = parseDate(a.period);
            const dateB = parseDate(b.period);
            return sortOrder === 'newest'
                ? dateB.getTime() - dateA.getTime()
                : dateA.getTime() - dateB.getTime();
        });

    const categories: { key: Category; label: string; icon: React.ReactNode }[] = [
        { key: 'all', label: 'All', icon: null },
        { key: 'PlanningAssistant', label: '實習', icon: <Briefcase size={16} /> },
        { key: 'FullTime', label: '全職', icon: <GraduationCap size={16} /> },
    ];

    // 渲染工作經驗頁面
    return (
        <>
            {/* Controls - 控制項 */}
            <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4 mb-6">
                {/* Category Tabs - 分類選項 */}
                <div className="flex items-center gap-2 bg-slate-100 rounded-lg p-1 border border-slate-200">
                    {categories.map((cat) => (
                        <button
                            key={cat.key}
                            onClick={() => setCategory(cat.key)}
                            className={`flex items-center gap-1.5 px-3 py-2 rounded-md transition-colors text-sm font-medium ${category === cat.key
                                ? 'bg-white text-blue-600 shadow-sm'
                                : 'text-slate-600 hover:text-blue-600 hover:bg-slate-200'
                                }`}
                        >
                            {cat.icon}
                            {cat.label}
                        </button>
                    ))}
                </div>

                {/* Sort Button - 排序按鈕 */}
                <button
                    onClick={() => setSortOrder(prev => prev === 'newest' ? 'oldest' : 'newest')}
                    className="flex items-center gap-2 px-4 py-2 bg-white hover:bg-slate-50 text-slate-600 rounded-lg transition-colors text-sm font-medium border border-slate-200 shadow-sm"
                >
                    <Calendar size={18} />
                    {sortOrder === 'newest' ? 'Newest First' : 'Oldest First'}
                </button>
            </div>

            {/* Timeline View - 時間軸視圖 */}
            <div className="relative">
                {/* Timeline Line - 時間軸線(置中對齊圓圈) */}
                <div className="absolute left-[15px] md:left-[31px] top-0 bottom-0 w-1 md:w-1.5 bg-gradient-to-b from-blue-500 via-blue-300 to-slate-200" />

                <div className="space-y-6">
                    {filteredWorks.map((work) => {
                        const isExpanded = expandedWork === work.id;
                        // 取得年份
                        const year = work.period.split(' ')[1] || work.period.split(' ')[0];

                        return (
                            <div key={work.id} className="relative flex gap-4 md:gap-6">
                                {/* Timeline Dot - 時間軸圓圈(蓋住線) */}
                                <div className="relative z-10 flex-shrink-0">
                                    <div className={`w-8 h-8 md:w-16 md:h-16 rounded-full border-4 flex items-center justify-center bg-white shadow-sm ${work.type === 'public'
                                        ? 'border-green-500'
                                        : 'border-orange-500'
                                        }`}>
                                        <span className="text-[10px] md:text-sm font-bold text-slate-700">
                                            {year}
                                        </span>
                                    </div>
                                </div>

                                {/* Content Card - 內容卡片 */}
                                <div className="flex-1 bg-white rounded-lg border border-slate-200 overflow-hidden shadow-sm">
                                    {/* Header - Always Visible */}
                                    <button
                                        onClick={() => toggleExpand(work.id)}
                                        className="w-full p-4 md:p-6 text-left hover:bg-slate-50 transition-colors"
                                    >
                                        <div className="flex flex-col md:flex-row md:items-start md:justify-between gap-2 mb-3">
                                            <div className="flex-1">
                                                {work.organization && (
                                                    <p className="text-blue-600 font-medium text-sm flex items-center gap-1.5 mb-1">
                                                        <Building2 size={14} />
                                                        {work.organization}
                                                    </p>
                                                )}
                                                <h3 className="text-lg md:text-xl font-bold text-slate-900">
                                                    {work.title}
                                                </h3>
                                            </div>
                                            <div className="flex items-center gap-3 flex-shrink-0">
                                                <span className={`px-2 py-0.5 rounded-full text-xs font-semibold border ${work.type === 'public'
                                                    ? 'bg-green-100 text-green-700 border-green-200'
                                                    : 'bg-orange-100 text-orange-700 border-orange-200'
                                                    }`}>
                                                    {work.type === 'public' ? 'Public' : 'Private'}
                                                </span>
                                                <ChevronDown
                                                    size={20}
                                                    className={`text-slate-400 transition-transform duration-300 ${isExpanded ? 'rotate-180' : ''
                                                        }`}
                                                />
                                            </div>
                                        </div>

                                        {/* Role - 角色 */}
                                        {work.role && (
                                            <p className="text-slate-400 text-sm mb-2">{work.role}</p>
                                        )}

                                        <p className="text-slate-500 text-xs flex items-center gap-1.5 mb-3">
                                            <Calendar size={12} />
                                            {work.period}
                                        </p>

                                        <p className="text-slate-600 text-sm leading-relaxed">
                                            {work.summary}
                                        </p>
                                    </button>

                                    {/* Expanded Content - 展開內容 */}
                                    {isExpanded && (
                                        <div className="border-t border-slate-200 p-4 md:p-6 space-y-6">
                                            {/* Screenshot - 截圖 */}
                                            {work.screenshot && (
                                                <div className="bg-slate-500 rounded-lg border border-slate-600 overflow-hidden">
                                                    <img
                                                        src={work.screenshot}
                                                        alt={`${work.title} screenshot`}
                                                        className="w-full h-auto object-cover"
                                                    />
                                                </div>
                                            )}
                                            {work.screenshot_2 && (
                                                <div className="bg-slate-500 rounded-lg border border-slate-600 overflow-hidden">
                                                    <img
                                                        src={work.screenshot_2}
                                                        alt={`${work.title} screenshot`}
                                                        className="w-full h-auto object-cover"
                                                    />
                                                </div>
                                            )}

                                            {/* Description - 描述 */}
                                            <div>
                                                <h4 className="text-lg font-bold text-blue-700 mb-2">Overview</h4>
                                                <p className="text-slate-600 text-sm leading-relaxed">{work.description}</p>
                                            </div>

                                            {/* Highlights - 重點 */}
                                            <div>
                                                <h4 className="text-lg font-bold text-blue-700 mb-3">Highlights</h4>
                                                <ul className="text-slate-600 text-sm space-y-2">
                                                    {work.highlights.map((highlight, idx) => (
                                                        <li key={idx} className="flex items-start gap-2">
                                                            <span className="text-blue-400 mt-0.5 flex-shrink-0">→</span>
                                                            <span>{highlight}</span>
                                                        </li>
                                                    ))}
                                                </ul>
                                            </div>

                                            {/* Technologies - 技術棧 */}
                                            <div>
                                                <h4 className="text-lg font-bold text-blue-700 mb-3">Tech Stack</h4>
                                                <div className="flex flex-wrap gap-2">
                                                    {work.tech.map(tech => (
                                                        <span
                                                            key={tech}
                                                            className="px-3 py-1.5 bg-blue-50 text-blue-700 rounded-lg text-sm font-medium border border-blue-200"
                                                        >
                                                            {tech}
                                                        </span>
                                                    ))}
                                                </div>
                                            </div>

                                            {/* Challenges - 挑戰 */}
                                            {work.challenges.length > 0 && (
                                                <div>
                                                    <h4 className="text-lg font-bold text-blue-700 mb-3">Challenges & Solutions</h4>
                                                    <div className="space-y-3">
                                                        {work.challenges.map((item, idx) => (
                                                            <div key={idx} className="bg-slate-50 rounded-lg p-4 border border-slate-200">
                                                                <h5 className="text-sm font-semibold text-orange-600 mb-2">
                                                                    🔥 {item.challenge}
                                                                </h5>
                                                                <p className="text-slate-600 text-sm">
                                                                    <span className="text-green-300 font-semibold">✓</span> {item.solution}
                                                                </p>
                                                            </div>
                                                        ))}
                                                    </div>
                                                </div>
                                            )}

                                            {/* GitHub Link - GitHub 連結 */}
                                            {work.type === 'public' && work.github && (
                                                <div className="pt-2">
                                                    <a
                                                        href={work.github}
                                                        target="_blank"
                                                        rel="noopener noreferrer"
                                                        className="inline-flex items-center gap-2 px-4 py-2 bg-slate-100 hover:bg-slate-200 rounded-lg text-blue-600 text-sm font-medium transition-colors"
                                                    >
                                                        <Github size={18} />
                                                        查看原始碼
                                                    </a>
                                                </div>
                                            )}

                                            {/* Private Notice - 私人公告 */}
                                            {work.type === 'private' && (
                                                <div className="flex items-center gap-2 text-orange-300 bg-orange-500/10 border border-orange-500/30 rounded-lg px-4 py-3">
                                                    <span>🔒</span>
                                                    <span className="text-sm">This is a private project. Source code is not publicly available.</span>
                                                </div>
                                            )}
                                        </div>
                                    )}
                                </div>
                            </div>
                        );
                    })}
                </div>
            </div>

            {/* Empty State */}
            {filteredWorks.length === 0 && (
                <div className="text-center py-12 text-slate-500">
                    <p>No works found in this category.</p>
                </div>
            )}
        </>
    );
}