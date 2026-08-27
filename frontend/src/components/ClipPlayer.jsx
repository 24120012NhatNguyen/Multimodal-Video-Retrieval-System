import React, { useEffect, useRef, useState } from "react";
import { media_url } from "../helper/web_url.js";

// Clip ngắn [pts − w, pts + w] quanh một keyframe.
//
// Một ảnh tĩnh không phân biệt được "đang bước vào" với "đang bước ra", "xe
// đang chạy tới" với "xe đang lùi". Bốn giây quanh frame là thao tác nhanh nhất
// để loại kết quả sai.
//
// Server phải là media_url (LOCAL): clip cắt từ data/videos, Kaggle không mount
// thư mục đó.

function ClipPlayer({ videoId, frameIdx, window: win = 2, autoPlay = true }) {
  const ref = useRef(null);
  const [err, setErr] = useState(null);
  const [w, setW] = useState(win);

  const pad = String(frameIdx).padStart(6, "0");
  const url = `${media_url}/clip/${videoId}/${pad}.mp4?window=${w}`;

  useEffect(() => {
    setErr(null);
    if (ref.current) ref.current.load();
  }, [url]);

  if (!videoId || frameIdx === undefined || frameIdx === null) return null;

  return (
    <div className="flex flex-col gap-1">
      <video
        ref={ref}
        src={url}
        controls
        autoPlay={autoPlay}
        loop
        muted={false}
        onError={() =>
          setErr(
            "Không tải được clip. Server media (socket_app.py) phải đang chạy " +
              "và data/videos phải có file gốc."
          )
        }
        className="w-full max-h-[360px] rounded-md bg-black"
      />
      <div className="flex items-center gap-1 text-xs text-slate-400">
        <span>
          {videoId} · frame {frameIdx} · ±{w}s
        </span>
        <span className="ml-auto">Độ rộng:</span>
        {[1, 2, 4, 8].map((v) => (
          <button
            key={v}
            type="button"
            onClick={() => setW(v)}
            className={`px-1.5 py-0.5 rounded transition ${
              w === v ? "bg-amber-600 text-white" : "bg-slate-700 hover:bg-slate-600"
            }`}
          >
            ±{v}s
          </button>
        ))}
      </div>
      {err && <div className="text-xs text-rose-400">{err}</div>}
    </div>
  );
}

export default ClipPlayer;
