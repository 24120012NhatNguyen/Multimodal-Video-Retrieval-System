import React from "react";

// Công tắc từng kênh bằng chứng, do NGƯỜI DÙNG gạt.
//
//   auto  hệ thống tự quyết theo loại truy vấn (đo được: truy vấn thị giác thì
//         mọi trọng số văn bản đều làm kết quả tệ đi, nên auto đặt chúng = 0)
//   on    ép bật, kể cả khi hệ thống cho là kênh này không có tín hiệu
//   off   ép tắt
//
// Người ngồi trước màn hình nhìn thấy kết quả, bộ phân loại thì không — nên ý
// người dùng thắng.

const CHANNELS = [
  ["siglip", "Hình ảnh", "SigLIP — nội dung nhìn thấy trong khung hình"],
  ["meta", "Metadata", "Tiêu đề / mô tả / từ khoá của video"],
  ["asr", "ASR", "Lời nói trong video (Whisper)"],
  ["ocr", "OCR", "Chữ hiện trên màn hình"],
];

const MODES = [
  ["auto", "Tự"],
  ["on", "Bật"],
  ["off", "Tắt"],
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
    <div className="flex flex-wrap items-center gap-2 px-2 py-1 text-xs rounded-md bg-slate-900/80 text-slate-300">
      <span className="text-slate-500">Kênh bằng chứng:</span>
      {CHANNELS.map(([ch, label, hint]) => {
        const cur = m[ch] || "auto";
        const w = used ? used[ch] : undefined;
        return (
          <span key={ch} className="flex items-center gap-0.5" title={hint}>
            <span
              className={`mr-0.5 ${
                w === undefined
                  ? "text-slate-500"
                  : w > 0
                  ? "text-emerald-400"
                  : "text-slate-600 line-through"
              }`}
            >
              {label}
              {w !== undefined && w > 0 ? ` ×${w}` : ""}
            </span>
            {MODES.map(([v, t]) => (
              <button
                key={v}
                type="button"
                onClick={() => set(ch, v)}
                className={`px-1 py-0.5 rounded transition ${
                  cur === v
                    ? v === "off"
                      ? "bg-rose-700 text-white"
                      : v === "on"
                      ? "bg-emerald-700 text-white"
                      : "bg-slate-600 text-white"
                    : "bg-slate-800 hover:bg-slate-700"
                }`}
              >
                {t}
              </button>
            ))}
          </span>
        );
      })}
      {meta && meta.weightsNote && (
        <span className="text-amber-400" title={meta.weightsNote}>
          ⚠ đã lùi trọng số
        </span>
      )}
    </div>
  );
}

export default ChannelModes;
