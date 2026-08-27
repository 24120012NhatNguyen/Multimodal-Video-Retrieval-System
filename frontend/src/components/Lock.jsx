import React from "react";
import { AiOutlineLock, AiOutlineUnlock } from "react-icons/ai";

function Lock({ lock, setLock }) {
  return (
    <div
      id="lock"
      onClick={() => {
        setLock((old) => {
          if (old === true)
            document.getElementById("username").focus()
          return !old
        });
      }}
      title={lock ? "Bấm để sửa tên" : "Bấm để khoá tên lại"}
      className="absolute right-1.5 top-1/2 -translate-y-1/2 cursor-pointer p-0.5 rounded text-[color:var(--muted)] hover:text-[color:var(--accent)] transition">
      {lock ? (
        <AiOutlineLock className="w-4 h-4" />
      ) : (
        <AiOutlineUnlock className="w-4 h-4" />
      )}
    </div>
  );
}

export default Lock;
