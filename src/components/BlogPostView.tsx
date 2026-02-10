// src/components/BlogPostView.tsx
'use client';

import { useEffect, useState } from 'react';
import { ArrowLeft } from 'lucide-react';
import { Post } from '@/types';
import ReactMarkdown from 'react-markdown';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import 'katex/dist/katex.min.css';

interface Props {
    post: Post;
    onBack: () => void;
}

export default function BlogPostView({ post, onBack }: Props) {
    const [content, setContent] = useState('');
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        // 從 public 目錄 fetch markdown 檔案
        fetch(post.file)
            .then(res => res.text())
            .then(text => {
                setContent(text);
                setLoading(false);
            })
            .catch(err => {
                console.error('Failed to load post:', err);
                setLoading(false);
            });
    }, [post.file]);

    return (
        <div className="fixed inset-0 z-50 bg-gradient-to-br from-slate-50 via-white to-slate-100 overflow-y-auto">
            <div className="max-w-4xl mx-auto px-6 py-8">
                {/* 返回按鈕 */}
                <button
                    onClick={onBack}
                    className="flex items-center gap-2 text-slate-600 hover:text-blue-600 mb-8 transition-colors sticky top-4 bg-white/80 backdrop-blur-sm px-4 py-2 rounded-lg shadow-sm border border-slate-200"
                >
                    <ArrowLeft size={18} />
                    返回文章列表
                </button>

                {/* 文章標題 */}
                <h1 className="text-3xl font-bold mb-4 text-slate-900">{post.title}</h1>
                <p className="text-slate-500 mb-8">{post.date}</p>

                {/* 文章內容 */}
                {loading ? (
                    <p className="text-slate-500">載入中...</p>
                ) : (
                    <div className="prose prose-slate prose-lg max-w-none">
                        <ReactMarkdown
                            remarkPlugins={[remarkMath]}
                            rehypePlugins={[rehypeKatex]}
                        >
                            {content}
                        </ReactMarkdown>
                    </div>
                )}
            </div>
        </div>
    );
}