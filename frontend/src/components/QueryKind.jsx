import React from "react";

// Truy vấn này được xếp loại thế nào, và AI quyết định.
// Người dùng cần thấy được điều đó để biết khi nào nên can thiệp bằng tay.
const KIND_LABEL = {
  generic_chain: "Chuỗi hành động chung chung",
  anchored: "Có mỏ neo riêng biệt",
};

const SOURCE_LABEL = {
  nguoi_dung: "bạn chọn",
  llm: "LLM",
  heuristic: "tự dò",
};

function QueryKind({ meta, alignMode, setAlignMode }) {
  const q = meta && meta.query;

  return (
    <div className="flex flex-wrap items-center gap-1.5 px-2 py-1 text-xs rounded-md bg-slate-900/80 text-slate-300">
      <span className="text-slate-500">Dóng hàng thời gian:</span>
      {[
        ["auto", "Tự động"],
        ["on", "Bật"],
        ["off", "Tắt"],
      ].map(([v, label]) => (
        <button
          key={v}
          type="button"
          onClick={() => setAlignMode(v)}
          title={
            v === "auto"
              ? "Để hệ thống tự phân loại truy vấn"
              : v === "on"
              ? "Ép dóng hàng, kể cả khi hệ thống cho là không cần"
              : "Ép tắt, dùng tìm phẳng"
          }
          className={`px-1.5 py-0.5 rounded transition ${
            alignMode === v
              ? "bg-amber-600 text-white"
              : "bg-slate-700 hover:bg-slate-600"
          }`}
        >
          {label}
        </button>
      ))}

      {q && (
        <>
          <span className="mx-1 text-slate-600">|</span>
          <span
            title={q.kind_why || ""}
            className={`px-1.5 py-0.5 rounded ${
              q.kind === "generic_chain"
                ? "bg-sky-800 text-sky-100"
                : "bg-emerald-800 text-emerald-100"
            }`}
          >
            {KIND_LABEL[q.kind] || q.kind}
          </span>
          <span className="text-slate-500">
            ({SOURCE_LABEL[q.kind_source] || q.kind_source})
          </span>
          {q.anchors && q.anchors.length > 0 && (
            <span className="px-1.5 py-0.5 rounded bg-slate-700">
              mỏ neo: {q.anchors.join(", ")}
            </span>
          )}
          {meta.aligned && (
            <span className="px-1.5 py-0.5 rounded bg-amber-700 text-amber-100">
              đã dóng hàng
            </span>
          )}
          {!meta.aligned && meta.alignSkipped && (
            <span className="text-slate-500" title={meta.alignSkipped}>
              không dóng hàng
            </span>
          )}
        </>
      )}
    </div>
  );
}

export default QueryKind;
