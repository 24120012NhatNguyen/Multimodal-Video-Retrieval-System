import fs from "fs";
import path from "path";

// Pages Router API routes run on the Node.js runtime by default, so `fs` is
// available here even though it is not available in Edge middleware.
export const config = {
  api: {
    bodyParser: false,
  },
};

const PUBLIC_DIR = path.join(process.cwd(), "public");
const KEYFRAMES_PREFIX = "/static/images/Keyframes/";

// Given "/static/images/Keyframes/L26_c/V276/000265.jpg", try the path as-is,
// then with "_extract" added to (or removed from) the data_part segment only.
function resolveKeyframesPath(pathname) {
  if (!pathname || !pathname.startsWith(KEYFRAMES_PREFIX)) return null;

  const relative = pathname.slice(KEYFRAMES_PREFIX.length);
  const segments = relative.split("/").filter(Boolean);
  if (segments.length < 1) return null;

  const existsOnDisk = (segs) =>
    fs.existsSync(path.join(PUBLIC_DIR, "static/images/Keyframes", ...segs));

  if (existsOnDisk(segments)) {
    return pathname;
  }

  const [dataPart, ...rest] = segments;
  const altDataPart = dataPart.endsWith("_extract")
    ? dataPart.slice(0, -"_extract".length)
    : `${dataPart}_extract`;
  const altSegments = [altDataPart, ...rest];

  if (existsOnDisk(altSegments)) {
    return KEYFRAMES_PREFIX + altSegments.join("/");
  }

  return null;
}

export default function handler(req, res) {
  const { path: requestedPath } = req.query;
  if (typeof requestedPath !== "string") {
    res.status(400).json({ error: "Missing 'path' query param" });
    return;
  }

  const resolvedPath = resolveKeyframesPath(requestedPath);
  res.status(200).json({ resolvedPath });
}
