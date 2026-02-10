import type { NextConfig } from "next";

const repo = "personal_web_fang";
const isProd = process.env.NODE_ENV === "production";

const nextConfig: NextConfig = {
  output: "export",
  images: { unoptimized: true },
  trailingSlash: true,

  // GitHub Pages: https://hedgehog919.github.io/personal_web_fang/
  basePath: isProd ? `/${repo}` : undefined,
  assetPrefix: isProd ? `/${repo}/` : undefined,
};

export default nextConfig;
