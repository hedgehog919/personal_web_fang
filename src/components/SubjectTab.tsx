'use client';

import { useState } from 'react';
import { ArrowUpDown, GraduationCap, Award } from 'lucide-react';
import { subjectCategories } from '@/data/subjects';

export default function SubjectTab() {
  const [activeCategory, setActiveCategory] = useState<number>(0);
  const [sortBy, setSortBy] = useState<'grade' | 'score'>('grade');

  // 根據選擇的類別決定顯示的科目
  const currentSubjects = subjectCategories[activeCategory].subjects;

  // 年級排序順序
  const gradeOrder = { '大一': 6, '大二': 5, '大三': 4, '大四': 3, '碩一': 2, '碩二': 1 };

  // 排序科目
  const sortedSubjects = [...currentSubjects].sort((a, b) => {
    if (sortBy === 'grade') {
      return gradeOrder[a.grade] - gradeOrder[b.grade];
    } else {
      return b.score - a.score;
    }
  });

  // 計算成績填滿的格數
  const getFilledBars = (score: number) => {
    return Math.floor(score / 10);
  };

  // 計算各領域的平均分數和科目數量
  const calculateCategoryScores = () => {
    return subjectCategories.map(category => {
      const subjects = category.subjects;
      const avgScore = subjects.length > 0
        ? subjects.reduce((sum, s) => sum + s.score, 0) / subjects.length
        : 0;
      return {
        category: category.title,
        score: avgScore,
        count: subjects.length,
        subjects: subjects
      };
    });
  };

  const categoryScores = calculateCategoryScores();

  const toggleSort = () => {
    setSortBy(prev => prev === 'grade' ? 'score' : 'grade');
  };

  return (
    <div className="grid grid-cols-1 lg:grid-cols-[1fr_1.5fr] gap-6 h-full">
      {/* 左側：能力六角圖 */}
      <div className="bg-white rounded-lg p-6 border border-slate-200 flex flex-col h-[600px] shadow-sm">
        <h3 className="text-xl font-bold text-blue-700 mb-4 text-center">能力分布圖</h3>
        <div className="flex-1 flex items-center justify-center min-h-0">
          <svg
            width="100%"
            height="100%"
            viewBox="0 0 400 450"
            className="max-w-[400px] max-h-[450px]"
          >
            {/* 繪製背景網格 */}
            {[10, 20, 30, 40, 50, 60, 70, 80, 90, 100].map((level) => {
              const points = Array.from({ length: 6 }, (_, i) => {
                const angle = (Math.PI / 3) * i - Math.PI / 2;
                const radius = (level / 100) * 150;
                const x = 200 + radius * Math.cos(angle);
                const y = 200 + radius * Math.sin(angle);
                return `${x},${y}`;
              }).join(' ');

              return (
                <polygon
                  key={level}
                  points={points}
                  fill="none"
                  stroke={level === 100 ? '#94a3b8' : '#cbd5e1'}
                  strokeWidth={level === 100 ? '2' : '1'}
                  opacity={level === 100 ? '0.5' : '0.5'}
                />
              );
            })}

            {/* 繪製從中心到各頂點的線 */}
            {categoryScores.map((item, i) => {
              const angle = (Math.PI / 3) * i - Math.PI / 2;
              const x = 200 + 150 * Math.cos(angle);
              const y = 200 + 150 * Math.sin(angle);
              return (
                <line
                  key={i}
                  x1="200"
                  y1="200"
                  x2={x}
                  y2={y}
                  stroke="#cbd5e1"
                  strokeWidth="1"
                  opacity="0.5"
                />
              );
            })}

            {/* 繪製能力數據多邊形 */}
            <polygon
              points={categoryScores.map((item, i) => {
                const angle = (Math.PI / 3) * i - Math.PI / 2;
                const radius = (item.score / 100) * 150;
                const x = 200 + radius * Math.cos(angle);
                const y = 200 + radius * Math.sin(angle);
                return `${x},${y}`;
              }).join(' ')}
              fill="rgba(59, 130, 246, 0.3)"
              stroke="#3b82f6"
              strokeWidth="2"
            />

            {/* 繪製可點擊的數據點 */}
            {categoryScores.map((item, i) => {
              const angle = (Math.PI / 3) * i - Math.PI / 2;
              const radius = (item.score / 100) * 150;
              const x = 200 + radius * Math.cos(angle);
              const y = 200 + radius * Math.sin(angle);
              const isActive = activeCategory === i;

              return (
                <g key={i}>
                  <circle
                    cx={x}
                    cy={y}
                    r={isActive ? '8' : '5'}
                    fill={isActive ? '#3b82f6' : '#60a5fa'}
                    stroke="#fff"
                    strokeWidth="2"
                    style={{ cursor: 'pointer', transition: 'all 0.2s' }}
                    onClick={() => setActiveCategory(i)}
                  />
                  {/* 增加透明互動區域 */}
                  <circle
                    cx={x}
                    cy={y}
                    r="30"
                    fill="transparent"
                    style={{ cursor: 'pointer' }}
                    onClick={() => setActiveCategory(i)}
                  />
                </g>
              );
            })}

            {/* 標籤文字 */}
            {categoryScores.map((item, i) => {
              const angle = (Math.PI / 3) * i - Math.PI / 2;
              const labelRadius = 185;
              const x = 200 + labelRadius * Math.cos(angle);
              const y = 200 + labelRadius * Math.sin(angle);
              const isActive = activeCategory === i;

              return (
                <g
                  key={i}
                  style={{ cursor: 'pointer' }}
                  onClick={() => setActiveCategory(i)}
                >
                  <text
                    x={x}
                    y={y}
                    textAnchor="middle"
                    dominantBaseline="middle"
                    className={`font-semibold text-sm ${isActive ? 'fill-blue-700' : 'fill-slate-600'}`}
                  >
                    {item.category}
                  </text>
                  <text
                    x={x}
                    y={y + 18}
                    textAnchor="middle"
                    dominantBaseline="middle"
                    className={`font-bold text-base ${isActive ? 'fill-blue-600' : 'fill-slate-500'}`}
                  >
                    {item.score.toFixed(1)} ({item.count})
                  </text>
                </g>
              );
            })}
          </svg>
        </div>
      </div>

      {/* 右側：科目成績列表 */}
      <div className="flex flex-col space-y-4 h-[600px]">
        {/* 標題和排序按鈕 */}
        <div className="flex items-center justify-between flex-shrink-0">
          <h3 className="text-2xl font-bold text-blue-700">
            {subjectCategories[activeCategory].title}
          </h3>
          <button
            onClick={toggleSort}
            className="flex items-center gap-2 px-4 py-2 bg-white hover:bg-slate-50 text-slate-600 rounded-lg transition-colors text-sm font-medium border border-slate-200 shadow-sm"
          >
            {sortBy === 'grade' ? <GraduationCap size={18} /> : <Award size={18} />}
            {sortBy === 'grade' ? '按年級' : '按成績'}
            <ArrowUpDown size={16} />
          </button>
        </div>

        {/* 科目列表 - 可滾動 */}
        <div className="flex-1 overflow-y-auto space-y-3 pr-2">
          {sortedSubjects.map((subject) => {
            const filledBars = getFilledBars(subject.score);
            return (
              <div key={subject.name} className="bg-white rounded-lg p-4 border border-slate-200 shadow-sm">
                {/* Subject Name, Grade and Type */}
                <div className="flex justify-between items-center mb-3">
                  <h4 className="text-lg font-semibold text-slate-800">
                    {subject.name}
                  </h4>
                  <div className="flex items-center gap-2">
                    <span className="px-3 py-1 rounded-full text-xs font-semibold bg-purple-100 text-purple-700 border border-purple-200">
                      {subject.grade}
                    </span>
                    <span className={`px-3 py-1 rounded-full text-xs font-semibold ${subject.type === '必修'
                      ? 'bg-blue-100 text-blue-700 border border-blue-200'
                      : 'bg-green-100 text-green-700 border border-green-200'
                      }`}>
                      {subject.type}
                    </span>
                  </div>
                </div>

                {/* Score Bar and Number */}
                <div className="flex items-center gap-4">
                  {/* Progress Bar */}
                  <div className="flex-1 flex gap-1">
                    {[1, 2, 3, 4, 5, 6, 7, 8, 9, 10].map((bar) => (
                      <div
                        key={bar}
                        className={`flex-1 h-3 rounded-sm transition-all duration-300 ${bar <= filledBars
                          ? 'bg-blue-500'
                          : 'bg-slate-200'
                          }`}
                      />
                    ))}
                  </div>

                  {/* Score Number */}
                  <span className="text-2xl font-bold text-blue-600 min-w-[3rem] text-right">
                    {subject.score}
                  </span>
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}