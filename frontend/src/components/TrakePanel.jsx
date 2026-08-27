import React, { useMemo, useState } from "react";
import Image from "next/image";
import { web_url, apiHeaders, imageUrl } from "../helper/web_url.js";
import Button from "./ui/Button.jsx";
import Group from "./ui/Group.jsx";
import LoadingIcon from "./LoadingIcon.jsx";

// TRAKE — dóng hàng chuỗi khoảnh khắc.
//
// Nguyên tắc thiết kế: DP chọn ĐÚNG MỘT frame cho mỗi sự kiện, và khi hai sự
// kiện nhìn gần giống nhau thì nó chọn nhầm mà không có đường lùi. Đo trên
// eval/trake_ground_truth.json (sample_03): DP chọn frame 6672 cho sự kiện 1,
// mà 6672 lại nằm trong khoảng đáp án của sự kiện 2.
//
// Hệ này có NGƯỜI DÙNG ngồi trong. Nên mỗi sự kiện hiện cả một DÃY ứng viên xếp
// theo thời gian, lựa chọn của DP được đánh dấu — người dùng lướt qua và chốt
// bằng mắt, nhanh hơn nhiều so với gõ lại truy vấn.

// BTC đánh số sự kiện ngay trong câu: "(1) ... (2) ...". Cắt theo đúng mốc đó.
function splitEvents(text) {
  const parts = String(text || "").split(/\(\s*\d+\s*\)/);
  const ev = parts.slice(1).map((p) => p.trim().replace(/[,;.]+$/, "")).filter(Boolean);
  if (ev.length >= 2) return ev;
  return String(text || "")
    .split(/[;.]/)
    .map((p) => p.trim())
    .filter(Boolean);
}

function TrakePanel({ questionName, addView, onClose }) {
  const [text, setText] = useState("");
  const [videos, setVideos] = useState([]);
  const [loading, setLoading] = useState(false);
  const [note, setNote] = useState("");
  // Frame người dùng tự chốt cho từng sự kiện: {video_id: {k: frame_idx}}
  const [picked, setPicked] = useState({});

  const events = useMemo(() => splitEvents(text), [text]);

  const run = () => {
    if (events.length < 2) {
      alert(
        "TRAKE cần ít nhất 2 khoảnh khắc. Đánh số chúng trong câu, ví dụ:\n" +
          "(1) khuấy bột, (2) tráng bột lên chảo, (3) gập đôi bánh"
      );
      return;
    }
    setLoading(true);
    setNote("");
    fetch(`${web_url}/trake`, {
      method: "post",
      headers: apiHeaders(),
      body: JSON.stringify({
        query_vi: text,
        events: [],
        video_topn: 20,
        n_candidates: 12,
      }),
    })
      .then((r) => r.json())
      .then((d) => {
        if (d.error) {
          setNote(d.error);
          setVideos([]);
        } else {
          setVideos(d.videos || []);
          setNote(
            `${d.n_video} video · ${(d.events || []).length} sự kiện · ` +
              `chọn video bởi ${d.chon_video_boi}`
          );
        }
        setLoading(false);
      })
      .catch((e) => {
        setNote("Không gọi được /trake: " + e);
        setLoading(false);
      });
  };

  const pick = (vid, k, frame) =>
    setPicked((p) => ({ ...p, [vid]: { ...(p[vid] || {}), [k]: frame } }));

  const chosen = (v, k) => {
    const p = picked[v.video_id] || {};
    if (p[k] !== undefined) return p[k];
    const m = v.matched[k];
    return m ? m.frame_idx : null;
  };

  const submit = (v) => {
    if (!questionName) {
      alert("Chọn câu hỏi trước khi nộp dãy TRAKE");
      return;
    }
    const seq = v.matched.map((_, k) => chosen(v, k));
    if (seq.some((f) => f === null || f === undefined)) {
      alert("Còn sự kiện chưa có frame");
      return;
    }
    // Nộp dãy dưới dạng MỘT dòng đáp án: "video, f1, f2, ..." — đúng định dạng
    // TRAKE của BTC.
    addView(`${v.video_id}#${seq[0]}`, seq.join(","));
    setNote(`Đã thêm dãy ${v.video_id}: ${seq.join(", ")} vào câu ${questionName}`);
  };

  return (
    <div className="flex flex-col gap-2 p-3 h-full overflow-auto">
      <div className="flex items-center gap-2">
        <span className="grp__label">TRAKE — chuỗi khoảnh khắc</span>
        <Button size="sm" className="ml-auto" onClick={onClose}>
          Đóng
        </Button>
      </div>

      <div className="flex gap-2 items-start">
        <textarea
          value={text}
          onChange={(e) => setText(e.target.value)}
          placeholder={
            "Mô tả các khoảnh khắc theo THỨ TỰ, đánh số bằng (1) (2) (3)...\n" +
            "Ví dụ: Tìm 3 khoảnh khắc đổ bánh xèo: (1) khuấy bột màu vàng, " +
            "(2) tráng bột lên chảo nóng, (3) gập đôi bánh lại."
          }
          className="inp flex-auto h-[76px] py-2 leading-snug resize-none"
        />
        <div className="flex flex-col gap-1.5">
          <Button variant="accent" onClick={run} disabled={loading}>
            {loading ? "Đang dóng..." : "Dóng hàng"}
          </Button>
          <span className="chip justify-center">{events.length} sự kiện</span>
        </div>
      </div>

      {note && <div className="text-[12px] text-[color:var(--muted)]">{note}</div>}
      {loading && <LoadingIcon />}

      {videos.map((v) => (
        <div key={v.video_id} className="panel p-2 flex flex-col gap-2">
          <div className="flex items-center gap-2 text-[12px]">
            <span className="mono chip">{v.video_id}</span>
            <span className="text-[color:var(--muted)]">
              điểm dóng hàng {v.dp_score}
              {v.n_skipped > 0 && ` · ${v.n_skipped} sự kiện yếu`}
            </span>
            <Button size="sm" className="ml-auto" onClick={() => submit(v)}>
              Thêm dãy vào bài
            </Button>
          </div>

          {(v.candidates || v.matched.map(() => [])).map((cands, k) => {
            const cur = chosen(v, k);
            return (
              <div key={k} className="flex flex-col gap-1">
                <div className="flex items-center gap-2 text-[11.5px]">
                  <span className="chip chip--warn">Sự kiện {k + 1}</span>
                  <span className="text-[color:var(--ink-2)] truncate flex-auto">
                    {events[k] || ""}
                  </span>
                  <span className="mono text-[color:var(--muted)]">
                    đang chọn {cur}
                  </span>
                  {/* Nhích từng frame. Đo trên ground truth: mọi sự kiện đều có
                      ứng viên nằm trong 4 giây quanh đáp án, nhưng chỉ 43% rơi
                      ĐÚNG khoảng — vì keyframe thưa hơn khoảng đáp án. Nút này
                      là chỗ người dùng bù phần còn lại. Bài nộp nhận frame_id
                      bất kỳ nên frame nhích ra vẫn nộp được. */}
                  {[-50, -12, +12, +50].map((d) => (
                    <button
                      key={d}
                      type="button"
                      onClick={() => cur != null && pick(v.video_id, k, cur + d)}
                      title={`nhích ${d > 0 ? "+" : ""}${(d / 25).toFixed(1)}s`}
                      className="seg__opt mono !px-1.5"
                    >
                      {d > 0 ? `+${d}` : d}
                    </button>
                  ))}
                </div>
                <div className="flex gap-1 overflow-x-auto pb-1">
                  {cur != null && !cands.some((c) => c.frame_idx === cur) && (
                    <span className="relative shrink-0 rounded-md overflow-hidden border-2 border-[color:var(--accent)]">
                      <span className="relative block h-[68px] w-[121px] bg-[color:var(--bg-soft)]">
                        <Image
                          src={imageUrl(`/keyframe/${v.video_id}/${String(cur).padStart(6, "0")}.jpg`)}
                          alt={`frame ${cur}`}
                          fill={true}
                          sizes="121px"
                          className="object-cover"
                        />
                      </span>
                      <span className="mono absolute top-0.5 left-0.5 px-1 rounded text-[10px] bg-[color:var(--accent)] text-[color:var(--accent-ink)]">
                        {cur}
                      </span>
                    </span>
                  )}
                  {cands.map((c) => (
                    <button
                      key={c.frame_idx}
                      type="button"
                      onClick={() => pick(v.video_id, k, c.frame_idx)}
                      title={`frame ${c.frame_idx} · ${c.pts_time}s · điểm ${c.score}`}
                      className={`relative shrink-0 rounded-md overflow-hidden border-2 transition ${
                        c.frame_idx === cur
                          ? "border-[color:var(--accent)]"
                          : "border-transparent hover:border-[color:var(--line-2)]"
                      }`}
                    >
                      <span className="relative block h-[68px] w-[121px] bg-[color:var(--bg-soft)]">
                        <Image
                          src={imageUrl(c.path)}
                          alt={`frame ${c.frame_idx}`}
                          fill={true}
                          sizes="121px"
                          className="object-cover"
                        />
                      </span>
                      <span className="mono absolute top-0.5 left-0.5 px-1 rounded text-[10px] bg-[color:var(--bg)]/80">
                        {c.frame_idx}
                      </span>
                      {c.la_lua_chon_dp && (
                        <span className="absolute bottom-0.5 right-0.5 px-1 rounded text-[9px] bg-[color:var(--accent)] text-[color:var(--accent-ink)]">
                          DP
                        </span>
                      )}
                    </button>
                  ))}
                  {cands.length === 0 && (
                    <span className="text-[11px] text-[color:var(--muted)]">
                      không có ứng viên
                    </span>
                  )}
                </div>
              </div>
            );
          })}
        </div>
      ))}
    </div>
  );
}

export default TrakePanel;
