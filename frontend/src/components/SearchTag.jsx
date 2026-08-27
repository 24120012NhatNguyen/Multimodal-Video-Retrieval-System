import React, { useEffect, useMemo, useState } from "react";
import { words } from "../helper/words";

// Danh sách tag lấy từ DỮ LIỆU THẬT (GET /data), không phải words.js.
//
// Đo được: trong 1199 từ của words.js chỉ 22% có mặt trong dữ liệu object của
// BTC (OpenImages). 78% còn lại gõ vào là chắc chắn không ra gì, mà giao diện
// cũ không hề báo — người dùng tưởng "không có video nào khớp".
//
// `vocab` rỗng (chưa tải xong, hoặc backend chưa sẵn sàng) thì lùi về words.js
// để ô tìm tag không bị trắng.

function SearchTag({ addTag, vocab, counts }) {
  const list = useMemo(
    () => (vocab && vocab.length ? vocab : words),
    [vocab]
  );
  const isReal = !!(vocab && vocab.length);
  const [q, setQ] = useState("");

  const shown = useMemo(() => {
    const s = q.trim().toLowerCase();
    if (!s) return list.slice(0, 60);
    const pre = [], mid = [];
    for (const t of list) {
      const lo = t.toLowerCase();
      if (lo.startsWith(s)) pre.push(t);
      else if (lo.includes(s)) mid.push(t);
      if (pre.length + mid.length > 200) break;
    }
    return pre.concat(mid).slice(0, 60);
  }, [q, list]);

  useEffect(() => {
    setQ((v) => v);
  }, [list]);

  return (
    <div className="flex flex-wrap relative h-full w-full hover:ease-in-out transition-all ">
      <input
        type="search"
        placeholder={isReal ? "Tìm lớp object (dữ liệu thật)" : "Search for tags"}
        title={
          isReal
            ? "Danh sách lấy từ chính dữ liệu object của BTC — gõ ra là chắc chắn có"
            : "Chưa tải được /data — đang dùng danh sách chép cứng, phần lớn không có trong dữ liệu"
        }
        className="h-fit transition-all hover:drop-shadow-[0px_2px_1px_rgba(255,255,255,0.2)] placeholder:italic text-slate-300 text-lg w-full p-1 pl-4 rounded-full bg-slate-800"
        onChange={(e) => setQ(e.target.value)}
      ></input>
      {shown.length > 0 && (
        <div className="h-[75px] overflow-auto flex-wrap p-1 gap-1 bg-slate-800 text-white w-full rounded-md flex-auto flex">
          {shown.map((tag) => (
            <span
              key={tag}
              onClick={() => addTag(tag)}
              title={counts && counts[tag] ? `${counts[tag]} lần xuất hiện` : ""}
              className="h-fit relative cursor-pointer hover:ring-2 ring-slate-400 w-max bg-slate-700 p-0.5 rounded-md "
            >
              {tag.replace(/_/g, " ")}
            </span>
          ))}
        </div>
      )}
      {!isReal && (
        <div className="text-[10px] text-amber-500 w-full px-1">
          chưa tải được danh sách lớp từ backend
        </div>
      )}
    </div>
  );
}

export default SearchTag;
