import React, { useEffect } from 'react'
import {
  AiFillCaretLeft,
  AiFillCaretRight,
} from "react-icons/ai";

function PageButton({ totalPage, autoFetch, isFilter, showAutoFetch, page, setPage, DivID, autoIgnore, handleAutoIgnore }) {
  useEffect(() => {
    if (DivID === "images" && autoIgnore && !isFilter) {
      if ((page === totalPage - 3 || (0 < totalPage && totalPage < 3)) && autoIgnore)
        autoFetch();
      if (page > totalPage) {
        showAutoFetch();
      }
    }
  }, [page])

  return (
    <div className="flex flex-wrap items-center justify-center gap-1.5 relative w-full my-3">
      {
        // page > 0 &&
        <button
          className="btn btn--sm"
          disabled={page <= 0}
          onClick={() => {
            document.getElementById(DivID).scrollTop = 0;
            setPage(page - 1);
          }}
        >
          <AiFillCaretLeft />
        </button>
      }
      <div className="mono px-3 h-7 flex items-center rounded-md text-[12px]
                      bg-[color:var(--panel)] border border-[color:var(--line)] text-[color:var(--ink-2)]">
        Trang {page > totalPage ? totalPage : page}<span className="text-[color:var(--muted)]">/{totalPage}</span>
      </div>
      <button
        className="btn btn--sm"
        disabled={page >= totalPage}
        onClick={() => {
          document.getElementById(DivID).scrollTop = 0;
          if (autoIgnore) handleAutoIgnore(page);
          setPage(page + 1);
        }}
      >
        <AiFillCaretRight />
      </button>
    </div>
  );
}

export default PageButton