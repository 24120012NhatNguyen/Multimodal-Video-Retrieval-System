import React, { useState } from "react";
import { web_url, apiHeaders } from "../helper/web_url.js";
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
    headers: apiHeaders(),
  };

  const handleFetch = () => {
    if (!videoId || start === "" || end === "") {
      alert("Nhập đủ Mã video, Từ và Đến");
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
    <div className="w-full px-1 mt-1 mb-1">
      <div className="grp flex-nowrap gap-1.5">
        <span className="grp__label">
          Dải frame
        </span>
        <input
          type="text"
          placeholder="Mã video"
          value={videoId}
          onChange={(e) => setVideoId(e.target.value)}
          className="inp inp--sm w-28"
        />
        <input
          type="number"
          placeholder="Từ"
          value={start}
          onChange={(e) => setStart(e.target.value)}
          className="inp inp--sm inp--num w-16"
        />
        <input
          type="number"
          placeholder="Đến"
          value={end}
          onChange={(e) => setEnd(e.target.value)}
          className="inp inp--sm inp--num w-16"
        />
        <input
          type="text"
          placeholder="Tìm trong dải này..."
          value={textQuery}
          onChange={(e) => setTextQuery(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter") handleFetch();
          }}
          className="inp inp--sm flex-auto"
        />
        <button
          type="button"
          onClick={handleFetch}
          className="btn btn--sm shrink-0"
        >
          Xem
        </button>
      </div>

      {message && (
        <div className="text-[11.5px] text-[color:var(--accent)] px-1 py-1">{message}</div>
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
