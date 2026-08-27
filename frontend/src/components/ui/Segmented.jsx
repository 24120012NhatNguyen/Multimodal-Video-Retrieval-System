import React from "react";

// Điều khiển phân đoạn: một lựa chọn trong vài lựa chọn loại trừ nhau.
//
// Dùng cho mọi công tắc "tự / bật / tắt" trong ứng dụng. Trước đây mỗi chỗ tự
// viết một cụm nút với class riêng, nên cùng một khái niệm lại trông khác nhau
// ở ba nơi — người dùng phải học lại từng chỗ.
//
// options: [{ value, label, tone?: "on"|"off"|"accent", title? }]

function Segmented({ options, value, onChange, label, className = "" }) {
  return (
    <span className={`inline-flex items-center gap-1.5 ${className}`}>
      {label && <span className="grp__label">{label}</span>}
      <span className="seg" role="group" aria-label={label || undefined}>
        {options.map((o) => (
          <button
            key={o.value}
            type="button"
            className="seg__opt"
            data-tone={o.tone || "neutral"}
            aria-pressed={value === o.value}
            title={o.title}
            onClick={() => onChange(o.value)}
          >
            {o.label}
          </button>
        ))}
      </span>
    </span>
  );
}

export default Segmented;
