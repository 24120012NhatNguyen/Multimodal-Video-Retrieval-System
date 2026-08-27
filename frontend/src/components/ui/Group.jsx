import React from "react";

// Nhóm điều khiển có tên. Đây là thứ thanh công cụ cũ thiếu hẳn: 20 nút nằm
// cùng một hàng phẳng, cùng trọng số thị giác, không nhóm nào cả — nên không
// tìm ra nút nào cả.

function Group({ label, children, flush = false, className = "" }) {
  return (
    <div className={`grp ${flush ? "grp--flush" : ""} ${className}`}>
      {label && <span className="grp__label">{label}</span>}
      {children}
    </div>
  );
}

export default Group;
