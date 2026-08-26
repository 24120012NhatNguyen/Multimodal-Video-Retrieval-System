import React from "react";

// Thứ hạng của video ở từng kênh bằng chứng — đây là thứ trả lời câu
// "vì sao video này nổi lên", và là cách phân biệt hàng chục video trông
// giống hệt nhau trong cùng một pack.
const CHANNEL_LABEL = {
  siglip: "SigLIP",
  meta: "Metadata",
  meta_fold: "Metadata (không dấu)",
  asr: "Lời nói",
  ocr: "Chữ trên hình",
};

// Hạng càng nhỏ càng mạnh; không lọt top thì để xám.
function rankColor(rank) {
  if (rank === null || rank === undefined) return "bg-slate-700 text-slate-400";
  if (rank <= 3) return "bg-emerald-600 text-white";
  if (rank <= 10) return "bg-emerald-800 text-emerald-100";
  if (rank <= 30) return "bg-amber-700 text-amber-100";
  return "bg-slate-600 text-slate-200";
}

function ExplainBadge({ explain, why, rrfScore }) {
  if (!explain) return null;
  const entries = Object.entries(explain);
  if (entries.length === 0) return null;

  return (
    <div className="flex flex-wrap items-center gap-1 px-2 py-1 mb-1 text-xs rounded-md bg-slate-900/80">
      {rrfScore !== undefined && rrfScore !== null && (
        <span className="px-1.5 py-0.5 font-mono rounded bg-slate-700 text-slate-200">
          RRF {Number(rrfScore).toFixed(4)}
        </span>
      )}
      {entries.map(([name, rank]) => {
        const terms = why && why[name] ? why[name] : [];
        return (
          <span
            key={name}
            title={
              terms.length
                ? `Từ khớp: ${terms.join(", ")}`
                : "Video này không lọt top của kênh đó"
            }
            className={`px-1.5 py-0.5 rounded ${rankColor(rank)}`}
          >
            {CHANNEL_LABEL[name] || name}
            {": "}
            {rank === null || rank === undefined ? "—" : `#${rank}`}
          </span>
        );
      })}
    </div>
  );
}

export default ExplainBadge;
