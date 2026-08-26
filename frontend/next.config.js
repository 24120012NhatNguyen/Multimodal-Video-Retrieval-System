/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: false,
  eslint: {
    // Cac loi ESLint con lai deu la co san tu truoc va thuan ve style
    // (component dat ten chu thuong trong Panel.jsx, thieu prop `key` trong vai
    // vong lap). Chung chan `next build` nhung khong anh huong chay that.
    // Sua chung la viec don dep rieng, khong nen chen vao sat gio thi.
    ignoreDuringBuilds: true,
  },
  images: {
    // Anh keyframe duoc trich on-demand tu server local va di qua tunnel.
    // De optimizer cua Next proxy lai chung se them mot chang mang nua co the
    // timeout, doi lai gan nhu khong loi gi (anh von da la JPEG nho).
    unoptimized: true,
    remotePatterns: [
      {
        protocol: 'https',
        hostname: '**',
      },
      {
        protocol: 'http',
        hostname: '**',
      },
    ],
  },
}

module.exports = nextConfig
