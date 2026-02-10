'use client';

import { useState } from 'react';
import About from '@/components/Profile';
import WorksTab from '@/components/WorksTab';
import AboutTab from '@/components/AboutTab';
import SubjectTab from '@/components/SubjectTab';
import ResearchTab from '@/components/ResearchTab';
import BlogTab from '@/components/BlogTab';
import Footer from '@/components/Footer';

// type TabType = 'about' | 'works' | 'research' | 'subjects';
type TabType = 'about' | 'works' | 'research' | 'blog' | 'subjects';

export default function Home() {
  const [activeTab, setActiveTab] = useState<TabType>('about');

  const tabs: { key: TabType; label: string }[] = [
    { key: 'about', label: 'About' },
    { key: 'works', label: 'Works' }
    // { key: 'research', label: 'Research' },
    // { key: 'blog', label: 'Blog' },
    // { key: 'subjects', label: 'Subjects' }
  ];

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-white to-slate-100 text-slate-900">

      {/* Main Content */}
      <section className="max-w-[95%] mx-auto px-4 py-8">
        <div className="grid grid-cols-1 lg:grid-cols-[0.5fr_2.5fr] gap-8 min-h-[85vh]">
          {/* Left Side - About */}
          <About />

          {/* Right Side - Tabs and Content */}
          <div className="flex flex-col">
            {/* Tabs */}
            <div className="flex flex-wrap gap-4 md:gap-6 mb-8">
              {tabs.map((tab) => (
                <button
                  key={tab.key}
                  onClick={() => setActiveTab(tab.key)}
                  className={`px-6 py-3 md:px-8 md:py-4 rounded-lg text-sm md:text-base font-semibold transition-all shadow-sm ${activeTab === tab.key
                    ? 'bg-blue-600 text-white shadow-md'
                    : 'bg-white text-slate-600 hover:bg-slate-50 border border-slate-200'
                    }`}
                >
                  {tab.label}
                </button>
              ))}
            </div>

            {/* Content Area */}
            <div className="bg-white/80 backdrop-blur-xl p-6 md:p-8 rounded-lg border border-slate-200 shadow-sm flex-1">
              {activeTab === 'about' && <AboutTab />}
              {activeTab === 'works' && <WorksTab />}
              {activeTab === 'research' && <ResearchTab />}
              {activeTab === 'blog' && <BlogTab />}
              {activeTab === 'subjects' && <SubjectTab />}
            </div>
          </div>
        </div>
      </section>

      <Footer />
    </div>
  );
}