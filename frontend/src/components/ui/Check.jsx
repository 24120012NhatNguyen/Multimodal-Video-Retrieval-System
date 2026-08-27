import React from "react";

// Hộp kiểm dạng nhãn bấm được: cả cụm là vùng bấm, không phải mỗi ô vuông 5px.

function Check({ id, checked, onChange, children, hint, disabled }) {
  return (
    <label className="chk" htmlFor={id} title={hint}>
      <input
        id={id}
        type="checkbox"
        checked={checked}
        disabled={disabled}
        onChange={onChange}
      />
      <span>{children}</span>
    </label>
  );
}

export default Check;
