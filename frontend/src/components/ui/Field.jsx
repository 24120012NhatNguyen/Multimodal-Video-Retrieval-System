import React from "react";

// Ô nhập CÓ NHÃN.
//
// Các ô số trên thanh công cụ cũ rộng 24px và không có nhãn — "K", "range",
// "Space" chỉ là placeholder, biến mất ngay khi gõ số vào. Không ai nhớ nổi ô
// nào là ô nào. Nhãn luôn hiện, và `hint` giải thích ô đó làm gì.

function Field({ label, hint, className = "", inputClassName = "", ...rest }) {
  return (
    <label className={`fld ${className}`} title={hint}>
      <span className="fld__label">{label}</span>
      <input className={`inp ${inputClassName}`} {...rest} />
    </label>
  );
}

export default Field;
