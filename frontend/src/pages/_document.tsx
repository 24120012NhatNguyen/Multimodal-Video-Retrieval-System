import { Html, Head, Main, NextScript } from "next/document";
// import { Inter } from "next/font/google"

// const inter = Inter({ subsets: ["latin"] })

export default function Document() {
  return (
    <Html lang="en">
      <Head>
        <meta charSet="UTF-8"></meta>
      </Head>
      <title>US_SHAREHOLDER</title>
      <body className={`text-slate-100 min-h-screen bg-slate-950 bg-[radial-gradient(ellipse_at_top_right,_var(--tw-gradient-stops))] from-slate-900 via-slate-950 to-black`}>
        <Main />
        <NextScript />
      </body>
    </Html>
  );
}
