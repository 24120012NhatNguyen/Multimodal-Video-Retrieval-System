import React from 'react'
import {BiHide } from "react-icons/bi";



function VideoWrapper({ children, id,
  // lst_idxs,
  handleIgnore, 
  filterFB
  // isIgnored
}) {
  const styles = {
    flex: filterFB ? 'none' : ''
  }

  return (
    // ${isIgnored ? "backdrop-blur-lg" : ""}
    <div
      className={`relative w-full flex justify-start gap-1.5 py-1
    ${filterFB ? "overflow-x-auto overflow-y-clip h-min flex-none flex-nowrap" : "flex-auto"}
      `}
    >
      <div
        style={{ zIndex: 2 }}
        className="flex flex-col items-center justify-center gap-2 sticky top-0 left-0 h-24 w-[72px] my-auto shrink-0
                   bg-[color:var(--panel)] border border-[color:var(--line)] rounded-[8px]"
      >
        <span className="mono text-[11px] leading-tight text-center break-all px-1 text-[color:var(--accent)]">
          {`${id}`}
        </span>
        <button
          type="button"
          onClick={() => handleIgnore()}
          title="Loại cả video này khỏi kết quả"
          className="rounded-md p-1 border border-[color:var(--line-2)] text-[color:var(--muted)]
                     hover:border-[color:var(--bad)] hover:text-[color:var(--bad)]
                     hover:bg-[color:var(--bad-dim)] transition"
        >
          <BiHide className="w-4 h-4" />
        </button>
      </div>
      {/* className, KHÔNG phải classname: viết thường thì React bỏ qua thuộc
          tính, và khung ảnh chưa từng nhận được lớp flex nào cả. */}
      <div
        style={styles}
        className={`relative flex h-max gap-1 p-1
        ${filterFB ? "flex-nowrap flex-none" : "flex-wrap"}`}
      >
        {children}
      </div>
    </div>
  );
}

export default VideoWrapper