import { NextResponse } from "next/server";

const KEYFRAMES_PREFIX = "/static/images/Keyframes/";

// Ask the Node-runtime API route to resolve the real on-disk path (with or
// without the "_extract" suffix on the data_part segment). Edge middleware
// cannot use `fs` directly, so this internal fetch is the bridge.
async function resolveKeyframePath(request, keyframePath) {
  const resolveUrl = new URL("/api/resolve-keyframe", request.url);
  resolveUrl.searchParams.set("path", keyframePath);

  const res = await fetch(resolveUrl.toString());
  if (!res.ok) return null;

  const data = await res.json();
  return data.resolvedPath || null;
}

export async function middleware(request) {
  const { pathname, searchParams } = request.nextUrl;

  // Case 1: direct request, e.g. /static/images/Keyframes/L26_c/V276/000265.jpg
  if (pathname.startsWith(KEYFRAMES_PREFIX)) {
    const resolvedPath = await resolveKeyframePath(request, pathname);
    if (resolvedPath && resolvedPath !== pathname) {
      const url = request.nextUrl.clone();
      url.pathname = resolvedPath;
      return NextResponse.rewrite(url);
    }
    // Not found under either variant: let Next.js return a real 404.
    return NextResponse.next();
  }

  // Case 2: Next.js Image Optimizer, e.g. /_next/image?url=%2Fstatic%2F...&w=...&q=...
  if (pathname === "/_next/image") {
    const imageUrl = searchParams.get("url");
    if (imageUrl && imageUrl.startsWith(KEYFRAMES_PREFIX)) {
      const resolvedPath = await resolveKeyframePath(request, imageUrl);
      if (resolvedPath && resolvedPath !== imageUrl) {
        const url = request.nextUrl.clone();
        url.searchParams.set("url", resolvedPath);
        return NextResponse.rewrite(url);
      }
    }
    return NextResponse.next();
  }

  return NextResponse.next();
}

export const config = {
  matcher: ["/static/images/Keyframes/:path*", "/_next/image"],
};
