// 設定工作經驗的資料來源，
// 含 id、title、organization、role、period、category、type、summary、description、highlights、tech、challenges、screenshot、github 等欄位
import { WorkItem } from '@/types'; // 引入工作經驗的類型定義

export const works: WorkItem[] = [
    {
        id: 'concurrent-assistant-NSYSU',
        title: 'Laboratory Administrative Assistant',
        organization: '國立中山大學－醫學科技研究所',
        role: '兼任助理（實驗室行政）',
        period: 'Feb 2025 - Present',
        category: 'PlanningAssistant',
        type: 'private',
        summary: '於醫學科技研究所實驗室擔任兼任助理，負責經費報帳、行政文書處理與實驗室日常行政支援，確保研究與行政作業順利進行。',
        description: '本職位主要負責實驗室行政與文書相關作業，包含研究計畫經費報帳、單據整理與行政文件彙整。協助處理實驗室日常行政事務，並配合教師與研究人員需求進行資料整理與流程溝通，以確保行政作業符合作業規範並提升整體效率。',
        highlights: [
            '負責研究計畫與實驗室相關經費報帳作業，整理與核對各項單據資料',
            '處理實驗室行政與文書作業，包含文件整理、資料彙整與歸檔',
            '協助教師與研究人員處理行政流程，確保作業符合校內規範',
            '支援實驗室日常行政事務，維持研究與行政運作順暢'
        ],
        tech: [
            '經費報帳與行政流程處理',
            '文件管理與資料彙整',
            '實驗室行政事務支援',
            '跨單位溝通與流程協調'
        ],
        challenges: [],
        screenshot: '',
        screenshot_2: ''
    },
    {
        id: 'nurse-TCM',
        title: 'TCM Nurse',
        organization: '尼克科學中醫診所',
        role: '全職－中醫診所護理師',
        period: 'Mar 2024 - Sep 2024',
        category: 'FullTime',
        type: 'private',
        summary: '於中醫診所擔任護理師，負責門診護理作業、病患照護與診療流程協助，確保門診運作順暢並提升病患就醫體驗。',
        description: '本職位主要於中醫診所門診環境中執行護理相關工作，協助醫師進行診療流程，包含病患接待、基本健康評估、治療前後照護及相關護理紀錄。於治療過程中觀察病患反應並提供即時協助，同時負責診所日常護理作業與行政配合，確保醫療流程安全、有序。',
        highlights: [
            '執行門診護理作業：病患接待、基本健康評估與治療前後照護',
            '協助中醫師進行診療流程，確保門診作業順暢與安全',
            '建立並維護病患護理與治療相關紀錄，確保資訊正確完整',
            '於治療過程中觀察病患狀況並提供適當安撫與照護，提升就醫體驗'
        ],
        tech: [
            '門診臨床護理與健康評估',
            '病患照護與護理紀錄管理',
            '診療流程協助與現場支援',
            '病患溝通與心理支持'
        ],
        challenges: [],
        screenshot: '',
        screenshot_2: ''
    },
    {
        id: 'information-specialist',
        title: 'ERP & Information System Maintenance Specialist',
        organization: '萬通人力資源顧問股份有限公司',
        role: '全職－ERP 及表單資訊系統維護人員',
        period: 'Aug 2022 - Feb 2024',
        category: 'FullTime',
        type: 'private',
        summary: '負責公司內部 ERP 與電子化表單資訊系統之維護與管理，確保系統穩定運作、資料正確性，並協助使用單位提升作業效率。',
        description: '本職位主要負責公司內部 ERP 系統及各式電子化表單之日常維護與管理，包含系統功能支援、資料處理與流程優化。透過與各部門溝通實際需求，協助調整表單與作業流程，確保資訊系統能有效支援人力資源及行政作業，同時維持系統穩定性與資料一致性。',
        highlights: [
            '維護與管理 ERP 系統，確保人資及相關業務流程正常運作',
            '負責電子化表單系統之維護、調整與問題排除，提升內部作業效率',
            '協助使用單位進行系統操作支援與需求溝通，降低使用門檻與錯誤率',
            '處理系統資料整合與檢核，確保資料正確性與一致性'
        ],
        tech: [
            'ERP 系統維護與管理',
            '電子化表單設計與流程維護',
            '資料處理、轉換與整合',
            '使用者需求溝通與系統支援'
        ],
        challenges: [],
        screenshot: '',
        screenshot_2: ''
    },
    {
        id: 'activity-assistant-PHC',
        title: 'Long-term care group activity assistant',
        organization: '岡山區衛生所',
        role: '課程專案 - 長照組活動助理',
        period: 'Aug 2021 - Oct 2021',
        category: 'PlanningAssistant',
        type: 'private',
        // github: 'https://github.com/32844583/NCU_1112_Course_DiaryBot',
        summary: '開發長照團體活動助理，協助所內社區組學姊管理社區改造、長期照顧相關活動及準備資料',
        description: '此專案旨在協助負責社區組學姊舉辦社區相關活動。專案的核心功能主要是協助學姊準備活動所需相關資料及帶領長者活動時能夠在安全環境中參加團體活動減緩長者老化。',
        highlights: [
            '協助社區組學姊準備活動所需相關資料',
            '帶領長者活動時能夠在安全環境中參加團體活動減緩長者老化',
        ],
        tech: ['活動相關資料處理、轉換及整合工作', '預防保護之護理措施', '管理活動行事曆與會議協調安排', '護理指導及諮詢'],
        challenges: [],
        screenshot: '',
        screenshot_2: 'screenshots/activity-assistant_2.png'
    },
    {
        id: 'nurse',
        title: 'Nurse',
        organization: '慈航護理之家',
        role: '全職 - 護理師',
        period: 'Jul 2021 - Jul 2021',
        category: 'FullTime',
        type: 'private',
        summary: '於住宿型長照機構執行臨床護理照護，涵蓋健康評估、用藥與病況紀錄、治療流程追蹤及心理支持，確保住民照護安全與照護品質。',
        description: '於住宿型長期照護機構擔任護理師，依醫囑執行住民日常護理與健康監測，落實用藥管理與病況觀察紀錄，並操作必要醫療照護設備以追蹤治療進度。於照護與治療過程中提供心理支持與衛教，協助住民降低不適與焦慮，並與跨專業團隊協作，維持照護流程的安全性與連續性。',
        highlights: [
            '執行日常臨床護理：生命徵象監測、一般健康評估、協助服藥與衛生教育',
            '建立並維護護理紀錄：用藥記錄、病徵觀察與異常狀況回報，確保資訊可追溯',
            '操作與管理醫療照護設備，追蹤治療進度與服藥遵從性，降低照護風險',
            '提供心理支持與情緒安撫，協助住民面對治療不適並提升照護配合度'
        ],
        tech: ['臨床護理與健康評估', '用藥管理與護理紀錄', '醫療設備操作與治療追蹤', '住民照護與心理支持'],
        challenges: [],
        screenshot: '',
        screenshot_2: ''
    },
    {
        id: 'activity-assistant-SZMC',
        title: 'Long-term care group activity assistant',
        organization: '樹人醫護管理專科學校',
        role: '課程專案 - 長照團體活動助理',
        period: 'Feb 2017 - Sep 2021',
        category: 'PlanningAssistant',
        type: 'private',
        // github: 'https://github.com/32844583/NCU_1112_Course_DiaryBot',
        summary: '開發長照團體活動助理，協助長期照顧組老師管理長期照顧相關團體活動及準備資料',
        description: '此專案旨在協助長期照顧組老師舉辦社區相關團體活動。專案的核心功能主要是協助老師準備活動所需相關資料及帶領長者活動時能夠在安全環境中參加團體活動減緩長者老化。',
        highlights: [
            '協助長期照顧組老師準備活動所需相關資料',
            '帶領長者活動時能夠在安全環境中參加團體活動減緩長者老化',
        ],
        tech: ['活動相關資料處理、轉換及整合工作', '預防保護之護理措施', '管理活動行事曆與會議協調安排', '護理指導及諮詢'],
        challenges: [
            // {
            //     challenge: '缺乏繁體中文的情感分析模型',
            //     solution: '除了串接 Google 翻譯 API，也可以透過提示詞引導模型直接判斷分析情感。'
            // },
            // {
            //     challenge: 'AI 回覆模式缺乏一致性',
            //     solution: '透過精心設計固定的提示詞模板 (Prompt Template)，可以更有效地引導和約束大型語言模型的輸出，能讓 AI 小天使的回覆風格、語氣和內容結構都標準化，從而提供更優質、更穩定的使用者體驗。'
            // }
        ],
        screenshot: 'screenshots/activity-assistant.png',
        screenshot_2: ''
    }
];