## 前言

最近決定用 Next.js 重新打造自己的個人網站，這篇文章記錄了整個學習過程。

## 為什麼選擇 Next.js？

Next.js 相比純 React 有幾個優勢：

- **檔案式路由**：不需要額外設定 router
- **靜態生成**：適合個人網站這類內容較固定的專案
- **內建優化**：圖片、字型自動優化

## 環境設置
![專案結構圖](/posts/images/nextjs-structure.png)

首先，使用 create-next-app 建立專案：

```bash
npx create-next-app@latest my-portfolio
```

選擇 TypeScript 和 Tailwind CSS 來獲得更好的開發體驗。

## 專案結構

```
src/
├── app/
│   ├── page.tsx
│   └── layout.tsx
├── components/
└── data/
```

## 心得

Next.js 的學習曲線比想像中平緩，特別是有 React 基礎的話，很快就能上手。
