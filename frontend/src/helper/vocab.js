// Từ vựng lớp object LẤY TỪ DỮ LIỆU THẬT, không phải danh sách chép cứng.
//
// Bảng icon trong icons.js là 80 lớp COCO và words.js là 1199 từ mô tả cảnh.
// Dữ liệu object của BTC lại là OpenImages (545 lớp, viết hoa). Đo được:
// words.js chỉ khớp 22% — người dùng gõ tag rồi bấm Send và nhận về rỗng mà
// không có gì báo là tag đó không tồn tại trong dữ liệu.
//
// GET /data trả về đúng danh sách lớp có thật, kèm số lần xuất hiện.

import { web_url, apiHeaders } from "./web_url.js";

let cache = null;

export function fetchVocab() {
  if (cache) return cache;
  cache = fetch(`${web_url}/data`, { method: "get", headers: apiHeaders() })
    .then((r) => r.json())
    .then((d) => ({
      objects: (d && d.objects) || [],
      names: ((d && d.objects) || []).map((o) => o.ten),
      videos: (d && d.videos) || [],
      nVideo: (d && d.n_video) || 0,
      nKeyframe: (d && d.n_keyframe) || 0,
      indexOk: !!(d && d.object_index && d.object_index.kha_dung),
      indexError: d && d.object_index && d.object_index.error,
    }))
    .catch((e) => {
      // Hỏng thì trả rỗng chứ không ném — panel vẫn phải mở được.
      cache = null;
      return { objects: [], names: [], videos: [], nVideo: 0, nKeyframe: 0,
               indexOk: false, indexError: String(e) };
    });
  return cache;
}
