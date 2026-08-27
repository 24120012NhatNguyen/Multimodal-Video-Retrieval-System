import React from "react";
import Image from "next/image";

// Một lớp object kéo được vào khung hình.
//
// Bản trước: nhãn rộng đúng 40px nên "person" hiện thành "perso", "bicycle"
// thành "bicycl" — người dùng phải đoán. Nay nhãn rộng bằng cả ô, cắt gọn bằng
// dấu ba chấm và tên đầy đủ nằm trong tooltip.

function Icon({ handleCreate, type, color }) {
  return (
    <button
      onClick={() => handleCreate(type)}
      type="button"
      title={color ? `Màu ${type}` : `Thêm "${type}" vào khung hình`}
      className="flex flex-col items-center gap-0.5 w-[58px] p-1 rounded-md
                 border border-transparent hover:border-[color:var(--accent)]
                 hover:bg-[color:var(--panel-2)] transition"
    >
      <span className="relative w-9 h-9 rounded bg-[color:var(--panel-3)] overflow-hidden">
        <Image alt={type} src={`/icons/${type}.png`} fill={true} sizes="36px" className="object-contain p-0.5" />
      </span>
      {!color && (
        <span className="w-full text-[9.5px] leading-tight text-center truncate text-[color:var(--muted)]">
          {type}
        </span>
      )}
    </button>
  );
}

export default Icon;
