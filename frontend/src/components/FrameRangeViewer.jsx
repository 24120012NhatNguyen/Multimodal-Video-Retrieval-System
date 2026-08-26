import React, { useState } from "react";
import { web_url } from "../helper/web_url.js";
import ImageListVideoPanel from "./ImageListVideoPanel.jsx";
import VideoWrapper from "./VideoWrapper.jsx";
import LoadingIcon from "./LoadingIcon.jsx";

function FrameRangeViewer({
  handleKNN,
  toggleFullScreen,
  handleSelect,
  handleIgnore,
  getIgnoredImages,
  questionName,
  addView,
}) {
  const [videoId, setVideoId] = useState("");
  const [start, setStart] = useState("");
  const [end, setEnd] = useState("");
  const [textQuery, setTextQuery] = useState("");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [message, setMessage] = useState("");

  const fetchGetObj = {
    method: "get",
    headers: new Headers({
      "ngrok-skip-browser-warning": "69420",
    }),
  };

  const handleFetch = () => {
    if (!videoId || start === "" || end === "") {
      alert("Nhập đầy đủ Video ID, Start và End");
      return;
    }
    setLoading(true);
    setMessage("");
    setResult(null);

    let url = `${web_url}/framerange?video_id=${encodeURIComponent(videoId)}&start=${start}&end=${end}`;
    if (textQuery && textQuery.trim()) {
      url += `&text_query=${encodeURIComponent(textQuery)}`;
    }

    fetch(url, fetchGetObj)
      .then((res) => res.json())
      .then((data) => {
        if (data.error) {
          setMessage(data.error);
          setResult(null);
        } else {
          setResult(data);
          if (data.message) {
            setMessage(data.message);
          }
        }
        setLoading(false);
      })
      .catch((e) => {
        alert("Frame range fetch failed: " + e);
        setLoading(false);
      });
  };

  return (
    <div className="w-full mt-1 mb-1">
      <div className="flex items-center gap-1 px-1 py-0.5 bg-slate-800 rounded-md">
        <span className="text-slate-400 text-xs font-semibold shrink-0">
          FrameRange
        </span>
        <input
          type="text"
          placeholder="Video ID"
          value={videoId}
          onChange={(e) => setVideoId(e.target.value)}
          className="w-24 text-xs placeholder:italic text-slate-300 p-1 rounded-md bg-slate-700"
        />
        <input
          type="number"
          placeholder="Start"
          value={start}
          onChange={(e) => setStart(e.target.value)}
          className="w-14 text-xs placeholder:italic text-slate-300 p-1 rounded-md bg-slate-700"
        />
        <input
          type="number"
          placeholder="End"
          value={end}
          onChange={(e) => setEnd(e.target.value)}
          className="w-14 text-xs placeholder:italic text-slate-300 p-1 rounded-md bg-slate-700"
        />
        <input
          type="text"
          placeholder="Tìm trong dải này..."
          value={textQuery}
          onChange={(e) => setTextQuery(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter") handleFetch();
          }}
          className="flex-auto text-xs placeholder:italic text-slate-300 p-1 rounded-md bg-slate-700"
        />
        <button
          type="button"
          onClick={handleFetch}
          className="text-xs px-2 py-1 rounded-md bg-emerald-700 hover:bg-emerald-600 hover:ring-1 ring-emerald-400 transition text-slate-200 font-semibold shrink-0"
        >
          Xem
        </button>
      </div>

      {message && (
        <div className="text-xs text-amber-400 px-1 py-0.5">{message}</div>
      )}

      {loading && <LoadingIcon />}

      {!loading &&
        result &&
        result.video_info &&
        result.video_info.lst_keyframe_paths.length > 0 && (
          <div className="mt-1">
            <VideoWrapper
              id={result.video_id}
              handleIgnore={() =>
                handleIgnore(result.video_info.lst_idxs)
              }
            >
              {result.video_info.lst_keyframe_paths.map((path, index) => (
                <ImageListVideoPanel
                  key={result.video_info.lst_idxs[index]}
                  addView={addView}
                  imagepath={path}
                  questionName={questionName}
                  id={result.video_info.lst_idxs[index]}
                  id_show={result.video_info.lst_keyframe_idxs[index]}
                  handleKNN={handleKNN}
                  handleIgnore={handleIgnore}
                  isIgnored={getIgnoredImages(
                    result.video_info.lst_idxs[index]
                  )}
                  handleSelect={() =>
                    handleSelect(
                      result.video_info.lst_keyframe_idxs[index],
                      result.video_id
                    )
                  }
                  toggleFullScreen={() =>
                    toggleFullScreen({
                      imgpath: path,
                      id: result.video_info.lst_idxs[index],
                    })
                  }
                />
              ))}
            </VideoWrapper>
          </div>
        )}
    </div>
  );
}

export default FrameRangeViewer;
