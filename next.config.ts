import type { NextConfig } from "next";

const repo = "personal_web_fang";
const isGhPages = process.env.DEPLOY_TARGET === "gh-pages";

const nextConfig: NextConfig = {
  output: "export",
  images: { unoptimized: true },
  trailingSlash: true,

  // 只有部署到 GitHub Pages 才需要子路徑
  basePath: isGhPages ? `/${repo}` : undefined,
  assetPrefix: isGhPages ? `/${repo}/` : undefined,
};

export default nextConfig;
