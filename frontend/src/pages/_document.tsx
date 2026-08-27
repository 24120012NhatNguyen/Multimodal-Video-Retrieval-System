import { Html, Head, Main, NextScript } from "next/document";

export default function Document() {
  return (
    <Html lang="vi">
      <Head>
        <meta charSet="UTF-8" />
        {/* IBM Plex: phủ đủ dấu tiếng Việt, và bản Mono cho id/số liệu xếp cột. */}
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="" />
        <link
          rel="stylesheet"
          href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap"
        />
      </Head>
      <title>AIC 2026 · Truy vấn video</title>
      <body className="min-h-screen">
        <Main />
        <NextScript />
      </body>
    </Html>
  );
}
