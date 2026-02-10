// 專案的類型定義

export interface Challenge {
  challenge: string;
  solution: string;
}
// 工作經驗的類型定義
export interface WorkItem {
  id: string; // 工作經驗的唯一識別碼

  // 基本資訊
  title: string;           // 工作經驗的標題
  organization?: string;   // 所屬組織（公司/學校/實驗室）
  role?: string;           // 角色
  period: string;          // 時間區間

  // 分類
  category: 'PlanningAssistant' | 'FullTime';
  type: 'public' | 'private';

  // 內容
  summary: string;         // 簡短描述（卡片用）
  description: string;     // 詳細描述（Modal 用）
  highlights: string[];    // 重點/成果

  // 技術與挑戰
  tech: string[];
  challenges: Challenge[];

  // 連結與圖片
  screenshot?: string; // 截圖
  screenshot_2?: string; // 截圖
  github?: string; // GitHub 連結
  externalLink?: string; // 外部連結
}

// Experience 相關類型 - 經歷（時間軸/列表）的類型定義
export interface Experience {
  key: string; // 經歷唯一識別
  title: string; // 標題（組織/單位）
  subtitle: string; // 副標（角色與期間）
  points: string[]; // 重點條列
  techs: string[]; // 技術標籤
}

// Research 相關類型 - 研究經驗的類型定義
export interface ResearchSection {
  id: string; // 研究經驗的唯一識別碼
  title: string; // 研究經驗的標題
  content: string; // 直接寫內容
}

export interface Research {
  title: string; // 研究經驗的標題
  author: string; // 作者
  advisor: string; // 指導教授
  department: string; // 系所
  date: string; // 日期
  sections: ResearchSection[]; // 章節
}

// Blog 相關類型 - 部落格文章的類型定義
export interface Post {
  slug: string; // 文章的唯一識別碼
  title: string; // 文章的標題
  date: string; // 文章的日期
  summary: string; // 文章的摘要
  tags: string[]; // 文章的標籤
  file: string;  // markdown 檔案路徑
}

// Blog 文章索引（不含 content） - 部落格文章的索引
export interface PostMeta {
  slug: string; // 文章的唯一識別碼
  title: string; // 文章的標題
  date: string; // 文章的日期
  summary: string; // 文章的摘要
  tags: string[]; // 文章的標籤
}


// 技能的類型定義
export interface Tool {
  name: string;
  icon?: string;
  proficiency: number;
}

// 技能分類的類型定義
export interface SkillCategory {
  title: string;
  tools: Tool[];
}

// 科目/課程的類型定義
export interface Subject {
  name: string;
  type: '必修' | '選修';
  score: number;
  grade: '大一' | '大二' | '大三' | '大四' | '碩一' | '碩二';
}

// 科目/課程分類的類型定義
export interface SubjectCategory {
  title: string;
  subjects: Subject[];
}