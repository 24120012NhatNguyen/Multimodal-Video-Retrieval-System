import React, { useState } from "react";
import { AiOutlineSelect, AiFillLike, AiFillDislike } from "react-icons/ai";
import { BsArrowsFullscreen, BsDatabaseAdd, BsQuestionCircle } from "react-icons/bs";
import { BiFileFind, BiSolidVideos, BiHide } from "react-icons/bi";
import Image from "next/image";
import { imageUrl } from "../helper/web_url.js";

// Thẻ một keyframe.
//
// Bản trước: bốn nút tròn trắng chỉ có icon, không nhãn — phải học thuộc mới
// biết nút nào làm gì. Và mỗi thẻ viền trắng dày, bốn góc gắn bốn ô xám đục
// che mất ảnh. Bản này: nhãn chữ trên mọi nút, viền tối, huy hiệu góc trong mờ.

function ActionButton({ id, onClick, icon, label, href, title }) {
  const cls =
    "flex flex-col items-center justify-center gap-0.5 w-[52px] h-[46px] rounded-md " +
    "bg-[color:var(--panel-2)]/95 border border-[color:var(--line-2)] " +
    "text-[color:var(--ink-2)] hover:text-[color:var(--ink)] " +
    "hover:border-[color:var(--accent)] hover:bg-[color:var(--panel-3)] transition";
  const inner = (
    <>
      {icon}
      <span className="text-[9.5px] leading-none font-medium">{label}</span>
    </>
  );
  if (href) {
    return (
      <a id={id} target="_blank" rel="noreferrer" href={href} className={cls} title={title || label}>
        {inner}
      </a>
    );
  }
  return (
    <button type="button" id={id} onClick={onClick} className={cls} title={title || label}>
      {inner}
    </button>
  );
}

function ImageList({
  imagepath,
  id,
  askVlm,
  handleKNN,
  handleSelect,
  toggleFullScreen,
  id_show,
  feedbackMode,
  handleFeedback,
  imgFeedback,
  handleIgnore,
  isIgnored,
  addView,
  questionName,
}) {
  const [answerText, setAnswerText] = useState("");
  const [asking, setAsking] = useState(false);

  // Hỏi VLM ngay trên frame này. Endpoint /qa đã có từ lâu nhưng giao diện chưa
  // từng gọi tới — người dùng phải tự nhìn ảnh rồi gõ tay câu trả lời.
  const ask = async () => {
    if (!askVlm) return;
    setAsking(true);
    try {
      const a = await askVlm(id);
      if (a) setAnswerText(a);
    } finally {
      setAsking(false);
    }
  };
  const showOverlay =
    feedbackMode && imgFeedback !== undefined ? "opacity-100" : "opacity-0 group-hover:opacity-100";

  return (
    <div
      className={`group relative shrink-0 rounded-lg overflow-hidden border transition
        ${
          isIgnored
            ? "border-[color:var(--bad)] opacity-45"
            : "border-[color:var(--line)] hover:border-[color:var(--accent)]"
        }`}
      key={id}
    >
      <div className="relative h-[120px] w-[213px] bg-[color:var(--bg-soft)]">
        <Image src={imageUrl(imagepath)} alt={`keyframe ${id_show}`} fill={true} className="object-cover" />

        {/* lớp phủ thao tác */}
        <div
          className={`absolute inset-0 flex flex-wrap gap-1 justify-center items-center
                      bg-[color:var(--bg)]/85 backdrop-blur-[2px] transition-opacity duration-200 ${showOverlay}`}
        >
          {feedbackMode ? (
            <>
              <button
                type="button"
                id={"like" + id}
                onClick={() => handleFeedback(id, "lst_pos_idxs")}
                title="Đánh dấu ảnh này ĐÚNG"
                className={`flex flex-col items-center gap-0.5 w-[62px] h-[52px] justify-center rounded-md border transition
                  ${
                    imgFeedback === "like"
                      ? "bg-[color:var(--good-dim)] border-[color:var(--good)] text-[color:var(--good)]"
                      : "bg-[color:var(--panel-2)] border-[color:var(--line-2)] text-[color:var(--ink-2)] hover:border-[color:var(--good)] hover:text-[color:var(--good)]"
                  }`}
              >
                <AiFillLike fontSize="1.25rem" />
                <span className="text-[10px] font-medium leading-none">Đúng</span>
              </button>
              <button
                type="button"
                id={"dislike" + id}
                onClick={() => handleFeedback(id, "lst_neg_idxs")}
                title="Đánh dấu ảnh này SAI"
                className={`flex flex-col items-center gap-0.5 w-[62px] h-[52px] justify-center rounded-md border transition
                  ${
                    imgFeedback === "dislike"
                      ? "bg-[color:var(--bad-dim)] border-[color:var(--bad)] text-[color:var(--bad)]"
                      : "bg-[color:var(--panel-2)] border-[color:var(--line-2)] text-[color:var(--ink-2)] hover:border-[color:var(--bad)] hover:text-[color:var(--bad)]"
                  }`}
              >
                <AiFillDislike fontSize="1.25rem" />
                <span className="text-[10px] font-medium leading-none">Sai</span>
              </button>
            </>
          ) : (
            <>
              <ActionButton
                id={"knn" + id}
                onClick={() => handleKNN(id)}
                icon={<BiFileFind fontSize="1.2rem" />}
                label="Ảnh giống"
                title="Tìm các keyframe giống ảnh này trên toàn corpus (KNN)"
              />
              <ActionButton
                id={"shot" + id}
                href={`shot?id=${id}&questionName=${questionName}`}
                icon={<BiSolidVideos fontSize="1.2rem" />}
                label="Cả video"
                title="Mở toàn bộ keyframe của video này ở tab mới"
              />
              <ActionButton
                id={"sendView" + id}
                onClick={() => addView(id, answerText)}
                icon={<BsDatabaseAdd fontSize="1.2rem" />}
                label="Thêm vào bài"
                title="Thêm frame này vào danh sách 100 dòng của câu hỏi đang chọn"
              />
              <ActionButton
                id={"select" + id}
                onClick={handleSelect}
                icon={<AiOutlineSelect fontSize="1.2rem" />}
                label="Nộp ngay"
                title="Nộp THẲNG frame này lên server BTC ngay lập tức — khác với Thêm vào bài"
              />
              {askVlm && (
                <ActionButton
                  id={"ask" + id}
                  onClick={ask}
                  icon={<BsQuestionCircle fontSize="1.2rem" />}
                  label={asking ? "..." : "Hỏi VLM"}
                  title="Hỏi Claude Haiku 4.5 trên frame này + OCR + lời nói, rồi điền vào ô đáp án"
                />
              )}
              <input
                type="text"
                placeholder="Đáp án cho câu Hỏi–Đáp"
                value={answerText}
                onClick={(e) => e.stopPropagation()}
                onChange={(e) => setAnswerText(e.target.value)}
                title="Chỉ dạng bài Hỏi–Đáp mới cần ô này"
                className="inp inp--sm w-[168px] h-7 text-[11px] px-2"
              />
            </>
          )}
        </div>
      </div>

      {/* huy hiệu góc: trong mờ, không che mất ảnh */}
      <span className="mono absolute top-1 left-1 px-1.5 py-0.5 rounded text-[11px] font-medium
                       bg-[color:var(--bg)]/75 text-[color:var(--ink)] pointer-events-none">
        {id_show}
      </span>

      <button
        type="button"
        onClick={() => handleIgnore(id)}
        title={isIgnored ? "Bỏ đánh dấu loại" : "Loại ảnh này khỏi kết quả"}
        className={`absolute top-1 right-1 p-1 rounded transition
          ${
            isIgnored
              ? "bg-[color:var(--bad-dim)] text-[color:var(--bad)]"
              : "bg-[color:var(--bg)]/75 text-[color:var(--muted)] opacity-0 group-hover:opacity-100 hover:text-[color:var(--bad)]"
          }`}
      >
        <BiHide className="w-4 h-4" />
      </button>

      <button
        type="button"
        onClick={toggleFullScreen}
        title="Xem to, kèm clip 4 giây quanh frame"
        className="absolute bottom-1 right-1 p-1.5 rounded bg-[color:var(--bg)]/75 text-[color:var(--muted)]
                   opacity-0 group-hover:opacity-100 hover:text-[color:var(--accent)] transition"
      >
        <BsArrowsFullscreen className="w-3.5 h-3.5" />
      </button>
    </div>
  );
}

export default ImageList;
