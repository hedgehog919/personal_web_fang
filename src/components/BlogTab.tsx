// BlogTab.tsx 部落格頁面
// 設定部落格頁面的資料來源，含 sortOrder、selectedTag、selectedPost 等欄位
'use client';

import { useState } from 'react';
import { Calendar, Tag, ArrowRight } from 'lucide-react';
import { posts } from '@/data/posts';
import { Post } from '@/types';
import BlogPostView from './BlogPostView';

type SortOrder = 'newest' | 'oldest';

export default function BlogTab() {
    const [sortOrder, setSortOrder] = useState<SortOrder>('newest');
    const [selectedTag, setSelectedTag] = useState<string | null>(null);
    const [selectedPost, setSelectedPost] = useState<Post | null>(null);  // 新增

    // 如果有選中的文章，顯示文章內容
    if (selectedPost) {
        return (
            <BlogPostView
                post={selectedPost}
                onBack={() => setSelectedPost(null)}
            />
        );
    }

    // 以下是原本的文章列表邏輯...
    const allTags = Array.from(new Set(posts.flatMap(post => post.tags)));

    const filteredPosts = posts
        .filter(post => !selectedTag || post.tags.includes(selectedTag))
        .sort((a, b) => {
            const dateA = new Date(a.date);
            const dateB = new Date(b.date);
            return sortOrder === 'newest'
                ? dateB.getTime() - dateA.getTime()
                : dateA.getTime() - dateB.getTime();
        });

    const formatDate = (dateStr: string) => {
        const date = new Date(dateStr);
        return date.toLocaleDateString('zh-TW', {
            year: 'numeric',
            month: 'long',
            day: 'numeric'
        });
    };

    return (
        <>
            {/* Intro Text */}
            <div className="mb-6 text-slate-600 leading-relaxed">
                這裡整理了我的學習紀錄，包括論文、前後端、雲端技術。
            </div>

            {/* Controls */}
            <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4 mb-6">
                {/* Tag Filter */}
                <div className="flex flex-wrap items-center gap-2">
                    <button
                        onClick={() => setSelectedTag(null)}
                        className={`px-3 py-1.5 rounded-full text-sm font-medium transition-colors ${!selectedTag
                            ? 'bg-blue-600 text-white'
                            : 'bg-white text-slate-600 hover:bg-slate-50 border border-slate-200'
                            }`}
                    >
                        All
                    </button>
                    {allTags.map(tag => (
                        <button
                            key={tag}
                            onClick={() => setSelectedTag(tag === selectedTag ? null : tag)}
                            className={`px-3 py-1.5 rounded-full text-sm font-medium transition-colors ${selectedTag === tag
                                ? 'bg-blue-600 text-white'
                                : 'bg-white text-slate-600 hover:bg-slate-50 border border-slate-200'
                                }`}
                        >
                            {tag}
                        </button>
                    ))}
                </div>

                {/* Sort Button */}
                <button
                    onClick={() => setSortOrder(prev => prev === 'newest' ? 'oldest' : 'newest')}
                    className="flex items-center gap-2 px-4 py-2 bg-white hover:bg-slate-50 text-slate-600 rounded-lg transition-colors text-sm font-medium border border-slate-200 shadow-sm"
                >
                    <Calendar size={18} />
                    {sortOrder === 'newest' ? 'Newest First' : 'Oldest First'}
                </button>
            </div>
            {/* Posts List */}
            <div className="space-y-4">
                {filteredPosts.map((post) => (
                    <button
                        key={post.slug}
                        onClick={() => setSelectedPost(post)}  // 改這裡
                        className="block w-full text-left bg-white rounded-lg border border-slate-200 p-6 hover:border-blue-500 transition-all group shadow-sm"
                    >
                        <div className="flex flex-col md:flex-row md:items-start md:justify-between gap-4">
                            <div className="flex-1">
                                <h3 className="text-xl font-bold text-slate-900 mb-2 group-hover:text-blue-600 transition-colors">
                                    {post.title}
                                </h3>
                                <p className="text-slate-500 text-sm mb-3 flex items-center gap-2">
                                    <Calendar size={14} />
                                    {formatDate(post.date)}
                                </p>
                                <p className="text-slate-600 text-sm leading-relaxed mb-4">
                                    {post.summary}
                                </p>
                                <div className="flex flex-wrap gap-2">
                                    {post.tags.map(tag => (
                                        <span
                                            key={tag}
                                            className="inline-flex items-center gap-1 px-2 py-1 bg-slate-100 text-slate-600 rounded text-xs"
                                        >
                                            <Tag size={10} />
                                            {tag}
                                        </span>
                                    ))}
                                </div>
                            </div>
                            <div className="flex items-center text-blue-600 text-sm font-medium group-hover:translate-x-1 transition-transform">
                                閱讀更多
                                <ArrowRight size={16} className="ml-1" />
                            </div>
                        </div>
                    </button>
                ))}
            </div>

            {/* Empty State */}
            {filteredPosts.length === 0 && (
                <div className="text-center py-12 text-slate-500">
                    <p>沒有找到相關文章</p>
                </div>
            )}
        </>
    );
}