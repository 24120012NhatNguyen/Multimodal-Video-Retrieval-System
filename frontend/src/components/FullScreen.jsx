import React from "react";
import { AiFillPlayCircle } from "react-icons/ai";
import Image from "next/image";
import { imageUrl } from "../helper/web_url.js";
import ClipPlayer from "./ClipPlayer.jsx";

// Xem một keyframe ở chế độ toàn màn hình, kèm BỐI CẢNH của nó.
//
// Trước đây chỗ này chỉ có ảnh tĩnh và một link YouTube. Ảnh tĩnh không phân
// biệt được "đang bước vào" với "đang bước ra" — mà đó chính là thứ quyết định
// đúng/sai của một đáp án. Nay có thêm:
//   · clip 4 giây quanh frame, cắt tại chỗ từ data/videos (server LOCAL)
//   · các keyframe liền trước/sau trên trục thời gian
//   · tiêu đề video + mốc giây để mở đúng chỗ trên YouTube

function FullScreen({ fullScreenImg, setFullScreenImg, relatedObj }) {
  if (fullScreenImg == null) return null;
  const rel = relatedObj || {};
  const near = rel.near_keyframes || [];
  const t0 = rel.video_range ? rel.video_range[0] : 0;

  return (
    <div
      onClick={() => setFullScreenImg(null)}
      className="fullscreenbackground justify-around w-screen h-screen bg-slate-950 flex absolute justify-center items-center bottom-auto right-auto rounded-md z-10"
    >
      <div
        onClick={(e) => e.stopPropagation()}
        className="p-2 rounded-md relative bg-slate-800 flex flex-col justify-center gap-1"
      >
        <div className="relative w-[860px] h-[484px] rounded-md">
          <Image
            src={imageUrl(fullScreenImg["imgpath"])}
            alt=""
            fill={true}
            className="rounded-md opacity-100"
          />
        </div>

        {rel.video_id !== undefined && (
          <div className="w-[860px]">
            <ClipPlayer
              videoId={rel.video_id}
              frameIdx={rel.frame_idx}
              window={rel.clip_window_sec || 2}
            />
          </div>
        )}

        <div className="w-[860px] flex items-center gap-2 text-xs text-slate-300">
          {rel.video_id && (
            <span className="px-1.5 py-0.5 rounded bg-slate-700">
              {rel.video_id} · frame {rel.frame_idx} ·{" "}
              {rel.pts_time !== undefined ? `${rel.pts_time}s` : ""}
            </span>
          )}
          {rel.title && (
            <span className="truncate flex-auto" title={rel.title}>
              {rel.title}
            </span>
          )}
          {rel.video_url && (
            <a
              href={`${rel.video_url}&t=${t0}s`}
              target="_blank"
              rel="noreferrer"
              title="Mở video gốc trên YouTube đúng mốc giây này"
              className="shrink-0 justify-center items-center flex h-8 w-8 rounded-full border border-black bg-orange-800 hover:bg-orange-600 transition-all"
            >
              <AiFillPlayCircle fontSize="1.2rem" />
            </a>
          )}
        </div>
      </div>

      <div
        onClick={(e) => e.stopPropagation()}
        className="related_img h-[800px] pt-1 flex flex-wrap justify-around items-start rounded-md bg-slate-700 overflow-auto w-[320px]"
      >
        <div className="w-full text-xs text-slate-300 px-2 py-1">
          Keyframe liền kề theo thời gian
        </div>
        {near.map((kf) => (
          <div
            key={kf.id}
            className={`m-0.5 p-0.5 rounded-md ${
              kf.la_anh_dang_xem ? "bg-amber-500" : "bg-slate-500"
            }`}
          >
            <div className="relative flex h-[169px] w-[300px]">
              <Image
                src={imageUrl(kf.imgpath)}
                alt=""
                fill={true}
                className="rounded-md"
              />
            </div>
            <div className="text-[10px] text-slate-100 text-center">
              {kf.frame_idx} · {kf.pts_time}s
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

export default FullScreen;
