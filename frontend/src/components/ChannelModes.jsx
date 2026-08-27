import React from "react";
import Segmented from "./ui/Segmented.jsx";

// Công tắc từng kênh bằng chứng, do NGƯỜI DÙNG gạt.
//
//   Tự   hệ thống tự quyết theo loại truy vấn (đo được: truy vấn thị giác thì
//        mọi trọng số văn bản đều làm kết quả tệ đi, nên Tự đặt chúng = 0)
//   Bật  ép bật, kể cả khi hệ thống cho là kênh này không có tín hiệu
//   Tắt  ép tắt
//
// Người ngồi trước màn hình nhìn thấy kết quả, bộ phân loại thì không — nên ý
// người dùng thắng.

const CHANNELS = [
  ["siglip", "Hình ảnh", "Nội dung nhìn thấy trong khung hình (SigLIP)"],
  ["meta", "Metadata", "Tiêu đề / mô tả / từ khoá của video"],
  ["asr", "Lời nói", "Tiếng nói trong video (ASR) — bật khi tên riêng hoặc con số chỉ được đọc lên"],
  ["ocr", "Chữ trên hình", "Chữ hiện trên màn hình — bật khi truy vấn nhắc tới biển hiệu, dòng chữ"],
];

const OPTS = [
  { value: "auto", label: "Tự" },
  { value: "on", label: "Bật", tone: "on" },
  { value: "off", label: "Tắt", tone: "off" },
];

function ChannelModes({ modes, setModes, meta }) {
  const m = modes || {};
  const used = (meta && meta.weightsUsed) || null;

  const set = (ch, v) => {
    const next = { ...m };
    if (v === "auto") delete next[ch];
    else next[ch] = v;
    setModes(next);
  };

  return (
    <div className="grp flex-wrap gap-x-3 gap-y-1.5">
      <span className="grp__label">Kênh bằng chứng</span>
      {CHANNELS.map(([ch, label, hint]) => {
        const w = used ? used[ch] : undefined;
        return (
          <span key={ch} className="inline-flex items-center gap-1.5" title={hint}>
            <span
              className={`text-[11.5px] ${
                w === undefined
                  ? "text-[color:var(--muted)]"
                  : w > 0
                  ? "text-[color:var(--good)]"
                  : "text-[color:var(--muted)] line-through"
              }`}
            >
              {label}
              {w > 0 && <span className="mono ml-1 opacity-80">×{w}</span>}
            </span>
            <Segmented
              value={m[ch] || "auto"}
              onChange={(v) => set(ch, v)}
              options={OPTS}
            />
          </span>
        );
      })}
      {meta && meta.weightsNote && (
        <span className="chip chip--warn" title={meta.weightsNote}>
          đã lùi trọng số
        </span>
      )}
    </div>
  );
}

export default ChannelModes;
