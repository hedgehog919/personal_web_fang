// 設定 blog 頁面
import { notFound } from 'next/navigation';
import { getPostBySlug, getAllPostSlugs } from '@/data/posts';
import fs from 'fs';
import path from 'path';
import ReactMarkdown from 'react-markdown';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';
import 'katex/dist/katex.min.css';

export function generateStaticParams() {
    return getAllPostSlugs().map((slug) => ({ slug }));
}

interface Props {
    params: Promise<{ slug: string }>;
}

export default async function BlogPostPage({ params }: Props) {
    const { slug } = await params;
    const post = getPostBySlug(slug);

    if (!post) {
        notFound();
    }

    const filePath = path.join(process.cwd(), 'public', post.file);
    let content = '';

    try {
        content = fs.readFileSync(filePath, 'utf-8');
    } catch (error) {
        console.error('Failed to read file:', error);
    }

    return (
        <div className="min-h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 text-white">
            <article className="max-w-3xl mx-auto px-4 py-12">
                <h1 className="text-3xl font-bold mb-4">{post.title}</h1>
                <p className="text-slate-400 mb-8">{post.date}</p>

                {/* prose 自動處理所有 Markdown 樣式 */}
                <div className="prose prose-invert prose-lg max-w-none">
                    <ReactMarkdown
                        remarkPlugins={[remarkMath]}
                        rehypePlugins={[rehypeKatex]}
                    >
                        {content}
                    </ReactMarkdown>
                </div>
            </article>
        </div>
    );
}