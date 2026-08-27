import React from "react";
import Segmented from "./ui/Segmented.jsx";

// Dóng hàng thời gian (DP) + hệ thống đang xếp truy vấn này vào loại nào.
// Người dùng cần thấy được điều đó để biết khi nào nên can thiệp bằng tay.

const KIND_LABEL = {
  generic_chain: "Chuỗi hành động chung",
  anchored: "Có mỏ neo riêng",
};

const SOURCE_LABEL = {
  nguoi_dung: "bạn chọn",
  llm: "LLM",
  heuristic: "tự dò",
};

function QueryKind({ meta, alignMode, setAlignMode }) {
  const q = meta && meta.query;

  return (
    <div className="grp flex-wrap gap-y-1.5">
      <Segmented
        label="Dóng hàng"
        value={alignMode}
        onChange={setAlignMode}
        options={[
          { value: "auto", label: "Tự",
            title: "Để hệ thống tự phân loại truy vấn" },
          { value: "on", label: "Bật", tone: "on",
            title: "Ép dóng hàng — dùng khi truy vấn là chuỗi hành động THEO THỨ TỰ mà từng cảnh rời đều tầm thường" },
          { value: "off", label: "Tắt", tone: "off",
            title: "Ép tắt, dùng tìm phẳng" },
        ]}
      />

      {q && (
        <>
          <span
            title={q.kind_why || ""}
            className={`chip ${q.kind === "anchored" ? "chip--on" : ""}`}
          >
            {KIND_LABEL[q.kind] || q.kind}
            <span className="text-[color:var(--muted)]">
              · {SOURCE_LABEL[q.kind_source] || q.kind_source}
            </span>
          </span>
          {q.anchors && q.anchors.length > 0 && (
            <span className="chip" title="Danh từ riêng hệ thống dò được">
              {q.anchors.join(", ")}
            </span>
          )}
          {meta.aligned && <span className="chip chip--warn">đã dóng hàng</span>}
          {!meta.aligned && meta.alignSkipped && (
            <span className="chip chip--off" title={meta.alignSkipped}>
              không dóng hàng
            </span>
          )}
        </>
      )}
    </div>
  );
}

export default QueryKind;
