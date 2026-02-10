import { Post } from '@/types';

export const posts: Post[] = [
    // {
    //     slug: 'Graph Neural Networks',
    //     title: 'Graph Neural Networks',
    //     date: '2025-07-23',
    //     summary: 'Graph Neural Network 的起源與原理',
    //     tags: ['GNN'],
    //     file: '/posts/graph-neural-network/index.md'
    // },
    {
        slug: '[Paper][Code] Stock Selection via Spatiotemporal Hypergraph Attention Network A Learning to Rank Approach',
        title: '[Paper][Code] Stock Selection via Spatiotemporal Hypergraph Attention Network A Learning to Rank Approach',
        date: '2026-01-05',
        summary: '以行業關聯股票形成圖結構，並經超圖卷積預測股票價格',
        tags: ['STHAN-SR', 'Industry', 'Time step attention', 'Relationship attention', 'Ranking', 'Medium', 'pytorch'],
        file: '/posts/stock-selection-hypergraph/index.md'
    },
    {
        slug: '[Paper][Code] Temporal Relational Ranking for Stock Prediction',
        title: '[Paper][Code] Temporal Relational Ranking for Stock Prediction',
        date: '2025-10-13',
        summary: '以顯式與隱式節點注意力衡量節點重要性，並經圖卷積預測股票價格',
        tags: ['RSR', 'TGC', 'Industry', 'Wiki relationship', 'Relationship attention', 'Ranking', 'Easy', 'tensorflow'],
        file: '/posts/temporal-relational-stock-ranking/index.md'
    }
];

// 取得所有文章 slugs（用於靜態生成）
export function getAllPostSlugs(): string[] {
    return posts.map(post => post.slug);
}

// 根據 slug 取得文章 metadata
export function getPostBySlug(slug: string): Post | undefined {
    return posts.find(post => post.slug === slug);
}